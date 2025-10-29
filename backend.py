"""
FastAPI Backend - Trích xuất dữ liệu từ Salesforce (Phiên bản Linh hoạt)
Hỗ trợ một endpoint /api/analyze đa mục đích.
"""

from fastapi import FastAPI, Query, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from simple_salesforce import Salesforce
from datetime import datetime
import warnings
import os
import numpy as np
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
import asyncio # Cần cho việc chạy pandas trong thread

warnings.filterwarnings('ignore')

app = FastAPI(title="Salesforce Contract Products API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cấu hình Salesforce
SALESFORCE_CONFIG = {
    'username': os.getenv('SALESFORCE_USERNAME'),
    'password': os.getenv('SALESFORCE_PASSWORD'),
    'security_token': os.getenv('SALESFORCE_SECURITY_TOKEN')
}

# --- Lớp SalesforceExporter (Không thay đổi) ---
# ... (Giữ nguyên toàn bộ class SalesforceExporter từ code gốc của bạn) ...
class SalesforceExporter:
    """Lớp kết nối và xuất dữ liệu từ Salesforce"""
    
    def __init__(self, username, password, security_token):
        self.username = username
        self.password = password
        self.security_token = security_token
        self.sf = None
        
    def connect(self):
        try:
            self.sf = Salesforce(
                username=self.username,
                password=self.password,
                security_token=self.security_token
            )
            return True
        except Exception as e:
            raise Exception(f"Lỗi kết nối Salesforce: {e}")
    
    def fetch_data(self):
        soql = """
         SELECT Name, 
           Contract__r.Account__r.Account_Code__c, 
           Product__r.STONE_Color_Type__c,
           Product__r.StockKeepingUnit,
           Product__r.Family,
           Segment__c,
           Contract__r.Created_Date__c,
           Contract__r.Name,
           Product_Discription__c,
           Length__c,
           Width__c,
           Height__c,
           Quantity__c,
           Crates__c,
           m2__c,
           m3__c,
           Tons__c, 
           Cont__c,
           Sales_Price__c,
           Charge_Unit_PI__c,
           Total_Price_USD__c
         FROM Contract_Product__c 
         ORDER BY Contract__r.Created_Date__c DESC
         """
        try:
            query_result = self.sf.query_all(soql)
            records = query_result['records']
            if not records: return None
            df = pd.json_normalize(records, sep='.')
            df = df.drop([col for col in df.columns if 'attributes' in col], axis=1)
            df.columns = df.columns.str.replace('Contract__r.', 'Contract_', regex=False)
            df.columns = df.columns.str.replace('Product__r.', 'Product_', regex=False)
            df.columns = df.columns.str.replace('Account__r.', 'Account_', regex=False)
            return df
        except Exception as e:
            raise Exception(f"Lỗi khi lấy dữ liệu: {e}")
    
    def transform_data(self, df):
        df_export = pd.DataFrame()
        df_export['Account Name: Account Code'] = df['Contract_Account_Account_Code__c']
        df_export['Product: STONE Color Type'] = df['Product_STONE_Color_Type__c']
        df_export['Product: Product SKU'] = df['Product_StockKeepingUnit']
        df_export['Contract Product Name'] = df['Name']
        df['Created_Date'] = pd.to_datetime(df['Contract_Created_Date__c'], errors='coerce')
        df_export['YEAR'] = df['Created_Date'].dt.year
        df_export['Product Discription'] = df['Product_Discription__c']
        df_export['Product: Mô tả sản phẩm'] = df['Product_Discription__c']
        df_export['Length'] = pd.to_numeric(df['Length__c'], errors='coerce')
        df_export['Width'] = pd.to_numeric(df['Width__c'], errors='coerce')
        df_export['Height'] = pd.to_numeric(df['Height__c'], errors='coerce')
        df_export['Quantity'] = pd.to_numeric(df['Quantity__c'], errors='coerce').fillna(0).astype(int)
        df_export['Crates'] = pd.to_numeric(df['Crates__c'], errors='coerce')
        df_export['m2'] = pd.to_numeric(df['m2__c'], errors='coerce')
        df_export['m3'] = pd.to_numeric(df['m3__c'], errors='coerce')
        df_export['Tons'] = pd.to_numeric(df['Tons__c'], errors='coerce')
        df_export['Cont'] = pd.to_numeric(df['Cont__c'], errors='coerce')
        df_export['Sales Price'] = pd.to_numeric(df['Sales_Price__c'], errors='coerce')
        df_export['Charge Unit (PI)'] = df['Charge_Unit_PI__c']
        df_export['Total Price (USD)'] = pd.to_numeric(df['Total_Price_USD__c'], errors='coerce')
        df_export['Product: Product Family'] = df['Product_Family']
        df_export['Segment'] = df['Segment__c']
        df_export['Contract Name'] = df['Contract_Name']
        df_export['Created Date (C)'] = df['Created_Date'].dt.strftime('%d/%m/%Y')
        df_export = df_export[(df_export['YEAR'] >= 2015) & (df_export['YEAR'] <= datetime.now().year)]
        df_export = df_export.dropna(subset=['Account Name: Account Code'])
        df_export = df_export.sort_values(by=['Account Name: Account Code', 'YEAR'], ascending=[True, False])
        df_export = df_export.reset_index(drop=True)
        return df_export

# --- HÀM HELPER CACHE (Không thay đổi) ---
# ... (Giữ nguyên hàm get_cached_transformed_data) ...
_global_data_cache = { "df": None, "last_fetch": None }
CACHE_DURATION_SECONDS = 600 # Cache trong 10 phút

async def get_cached_transformed_data() -> pd.DataFrame:
    now = datetime.now()
    if _global_data_cache["df"] is not None and _global_data_cache["last_fetch"] is not None:
        duration = (now - _global_data_cache["last_fetch"]).total_seconds()
        if duration < CACHE_DURATION_SECONDS:
            return _global_data_cache["df"]
    try:
        exporter = SalesforceExporter(**SALESFORCE_CONFIG)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, exporter.connect)
        df_raw = await loop.run_in_executor(None, exporter.fetch_data)
        if df_raw is None: return pd.DataFrame()
        df_export = await loop.run_in_executor(None, exporter.transform_data, df_raw)
        if df_export is None: return pd.DataFrame()
        _global_data_cache["df"] = df_export
        _global_data_cache["last_fetch"] = now
        return df_export
    except Exception as e:
        if _global_data_cache["df"] is not None:
            return _global_data_cache["df"]
        raise Exception(f"Lỗi nghiêm trọng khi fetch dữ liệu: {e}")

# --- THƯ VIỆN CÁC HÀM PHÂN TÍCH (LOGIC NGHIỆP VỤ) ---
# Đây là nơi chúng ta định nghĩa các "prompt" nghiệp vụ
# Lưu ý: Các hàm này là sync (không có async) vì Pandas là sync.
# Chúng sẽ được chạy trong thread pool.

def analysis_at_risk_customers(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Phân tích ví dụ của bạn: "Top N KH có doanh thu năm X nhưng không có đơn hàng từ năm Y"
    """
    revenue_year = params.get('revenue_year')
    no_order_since_year = params.get('no_order_since_year')
    top_n = params.get('top_n', 5)
    
    if not revenue_year or not no_order_since_year:
        raise ValueError("Thiếu 'revenue_year' hoặc 'no_order_since_year' trong params")

    current_year = datetime.now().year
    no_order_years = list(range(no_order_since_year, current_year + 1))
    
    # 1. Doanh thu năm target
    df_target = df[df['YEAR'] == revenue_year]
    if df_target.empty:
        return {"description": f"Không có doanh thu trong năm {revenue_year}", "data": []}
    revenue_by_customer = df_target.groupby('Account Name: Account Code')['Total Price (USD)'].sum()
    
    # 2. KH có đơn hàng gần đây
    df_recent = df[df['YEAR'].isin(no_order_years)]
    recent_customers_set = set(df_recent['Account Name: Account Code'].unique())
    
    # 3. Lọc (Set A - Set B)
    at_risk_customers_series = revenue_by_customer[~revenue_by_customer.index.isin(recent_customers_set)]
    
    # 4. Sắp xếp và lấy Top N
    top_at_risk = at_risk_customers_series.sort_values(ascending=False).head(top_n)
    
    # 5. Định dạng kết quả
    result_data = top_at_risk.reset_index().rename(columns={'Total Price (USD)': f'Revenue in {revenue_year}'}).to_dict(orient='records')
    
    return {
        "description": f"Top {len(result_data)} customers with revenue in {revenue_year} but no orders in {no_order_years}",
        "data": result_data
    }

def analysis_top_performers(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Một ví dụ phân tích khác: "Top N (KH, Sản phẩm, Segment,...) theo (Doanh thu, m2,...) trong năm Y"
    """
    group_by_col = params.get('group_by') # "Account Name: Account Code", "Product: Product Family"
    metric_col = params.get('metric', 'Total Price (USD)') # "Total Price (USD)", "m2"
    year = params.get('year')
    top_n = params.get('top_n', 5)
    
    if not group_by_col or not metric_col:
        raise ValueError("Thiếu 'group_by' hoặc 'metric' trong params")
    
    if group_by_col not in df.columns or metric_col not in df.columns:
        raise ValueError(f"Cột không hợp lệ: {group_by_col} hoặc {metric_col}")

    df_filtered = df.copy()
    if year:
        df_filtered = df_filtered[df_filtered['YEAR'] == year]
        
    if df_filtered.empty:
        return {"description": f"Không có dữ liệu cho năm {year}" if year else "Không có dữ liệu", "data": []}

    # 2. Group, Sum, Sort, Top N
    top_performers = df_filtered.groupby(group_by_col)[metric_col].sum() \
                                .sort_values(ascending=False) \
                                .head(top_n)
    
    # 3. Định dạng
    result_data = top_performers.reset_index().rename(columns={metric_col: f'Total {metric_col}'}).to_dict(orient='records')
    
    return {
        "description": f"Top {len(result_data)} {group_by_col} by {metric_col}" + (f" in {year}" if year else ""),
        "data": result_data
    }

# --- ENDPOINT CHÍNH (ĐA MỤC ĐÍCH) ---

# 1. Định nghĩa "Thư viện" các hàm phân tích
# Đây là mấu chốt: một "router" (bộ định tuyến) các hàm nghiệp vụ
ANALYSIS_REGISTRY = {
    "at_risk_customers": analysis_at_risk_customers,
    "top_performers": analysis_top_performers,
    # Bạn có thể thêm nhiều hàm nữa ở đây...
    # "new_customers": analysis_new_customers,
    # "product_trends": analysis_product_trends,
}

# 2. Định nghĩa "prompt" (payload) mà endpoint sẽ nhận
class AnalysisRequest(BaseModel):
    analysis_type: str  # Tên của hàm phân tích (VD: "at_risk_customers")
    params: Dict[str, Any] # Các tham số nghiệp vụ (VD: {"revenue_year": 2023, ...})

@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "ok",
        "message": "Salesforce Contract Products API (Phiên bản Linh hoạt)",
        "available_analyses": list(ANALYSIS_REGISTRY.keys())
    }

# *** ENDPOINT ĐA MỤC ĐÍCH MỚI ***
@app.post("/api/analyze")
async def handle_analysis_request(request: AnalysisRequest):
    """
    Endpoint linh hoạt: Nhận một "prompt" (AnalysisRequest)
    và trả về kết quả phân tích.
    """
    try:
        # 1. Tìm hàm phân tích trong thư viện
        analysis_function = ANALYSIS_REGISTRY.get(request.analysis_type)
        
        if not analysis_function:
            raise HTTPException(status_code=400, detail=f"Loại phân tích '{request.analysis_type}' không hợp lệ.")
        
        # 2. Lấy dữ liệu (từ cache hoặc fetch mới)
        df_data = await get_cached_transformed_data()
        
        if df_data.empty:
            return {"success": True, "analysis_type": request.analysis_type, "data": [], "message": "Không có dữ liệu để phân tích."}

        # 3. Chạy hàm phân tích (sync) trong thread pool (async)
        # Đây là cách đúng để chạy Pandas nặng trong FastAPI
        loop = asyncio.get_event_loop()
        result_data = await loop.run_in_executor(
            None,  # Sử dụng thread pool mặc định
            analysis_function, # Hàm sync (Pandas)
            df_data,           # Tham số 1 của hàm
            request.params     # Tham số 2 của hàm
        )

        return {
            "success": True,
            "analysis_type": request.analysis_type,
            **result_data # Gộp kết quả (description, data) vào
        }

    except ValueError as ve:
        # Lỗi từ logic nghiệp vụ (VD: thiếu param)
        raise HTTPException(status_code=422, detail=f"Lỗi tham số: {str(ve)}")
    except Exception as e:
        # Lỗi hệ thống
        raise HTTPException(status_code=500, detail=f"Lỗi máy chủ khi xử lý phân tích: {str(e)}")


# Endpoint cũ (vẫn hữu ích để tra cứu nhanh)
@app.get("/api/contract-products/by-account")
async def get_contract_details_by_account(
    account_code: str = Query(..., description="Account code to filter by (case-insensitive)")
):
    try:
        df_export = await get_cached_transformed_data()
        if df_export.empty:
            return {"success": True, "count": 0, "data": []}
        
        account_code_clean = account_code.strip().lower()
        filtered_df = df_export[
            df_export['Account Name: Account Code'].astype(str).str.strip().str.lower() == account_code_clean
        ]
        
        if filtered_df.empty:
            return {"success": True, "count": 0, "data": [], "message": f"No data found for account code '{account_code}'"}
        
        records = filtered_df.replace({np.nan: None}).to_dict(orient='records')
        return {"success": True, "count": len(records), "data": records, "account_code": account_code}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing data for account {account_code}: {str(e)}")

# (Không cần endpoint /api/contract-products nữa vì /api/analyze mạnh hơn)

if __name__ == "__main__":
    import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=8000)
    # Cần chạy theo cách này để có event loop cho run_in_executor
    loop = asyncio.get_event_loop()
    config = uvicorn.Config(app=app, loop=loop, host="0.0.0.0", port=8000)
    server = uvicorn.Server(config)
    loop.run_until_complete(server.serve())
