"""
FastAPI Backend - Trích xuất dữ liệu từ Salesforce
Đã cập nhật để hỗ trợ Phân Trang (Pagination) VÀ Lọc Động (Dynamic Filtering)
"""

from fastapi import FastAPI, Query, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from simple_salesforce import Salesforce
from datetime import datetime
import warnings
import os
import numpy as np

# MỚI: Thêm Pydantic và các kiểu dữ liệu
from pydantic import BaseModel
from typing import List, Optional, Literal

warnings.filterwarnings('ignore')

app = FastAPI(title="Salesforce Contract Products API")

# CORS middleware (Giữ nguyên)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cấu hình Salesforce (Giữ nguyên)
SALESFORCE_CONFIG = {
    'username': os.getenv('SALESFORCE_USERNAME'),
    'password': os.getenv('SALESFORCE_PASSWORD'),
    'security_token': os.getenv('SALESFORCE_SECURITY_TOKEN')
}

# =================================================================
# MỚI: Định nghĩa Model cho Request Lọc Động
# =================================================================

class FilterCondition(BaseModel):
    """Định nghĩa một điều kiện lọc đơn lẻ"""
    field: str  # Tên trường SOQL, ví dụ: 'Segment__c' hoặc 'Product__r.Family'
    operator: Literal["=", "!=", ">", "<", ">=", "<=", "LIKE"] = "="
    value: str | int | float | bool | None # Giá trị để so sánh

class FilterRequest(BaseModel):
    """
    Định nghĩa body cho request lọc.
    Sử dụng Body(...) thay vì Query(...) cho các tham số POST.
    """
    filters: List[FilterCondition] = []
    limit: int = Body(100, ge=1, le=500)
    offset: int = Body(0, ge=0)


# =================================================================
# THAY ĐỔI: Cập nhật Lớp SalesforceExporter
# =================================================================

class SalesforceExporter:
    """Lớp kết nối và xuất dữ liệu từ Salesforce"""
    
    def __init__(self, username, password, security_token):
        self.username = username
        self.password = password
        self.security_token = security_token
        self.sf = None
        
    def connect(self):
        """Kết nối tới Salesforce (Giữ nguyên)"""
        try:
            self.sf = Salesforce(
                username=self.username,
                password=self.password,
                security_token=self.security_token
            )
            return True
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Lỗi kết nối Salesforce: {e}")
    
    # THAY ĐỔI: 'fetch_data' đã được tổng quát hóa
    # Bỏ 'account_code' và thay bằng 'filters'
    def fetch_data(self, limit: int = 100, offset: int = 0, filters: Optional[List[FilterCondition]] = None):
        """Lấy dữ liệu từ Salesforce với phân trang và bộ lọc ĐỘNG"""
        
        where_clauses = []
        
        # Xây dựng mệnh đề WHERE một cách an toàn từ danh sách filters
        if filters:
            for f in filters:
                # 1. Kiểm tra toán tử hợp lệ (Pydantic đã làm nhưng kiểm tra lại)
                if f.operator not in ["=", "!=", ">", "<", ">=", "<=", "LIKE"]:
                    raise ValueError(f"Toán tử không hợp lệ: {f.operator}")

                # 2. Xử lý và chuẩn hóa giá trị
                sanitized_value = ""
                if f.value is None:
                    sanitized_value = "NULL"
                elif isinstance(f.value, str):
                    # Cực kỳ quan trọng: Thoát ký tự ' để chống SOQL Injection
                    sanitized_value = f"'{f.value.strip().replace("'", "\\'")}'"
                elif isinstance(f.value, (int, float)):
                    sanitized_value = str(f.value)
                elif isinstance(f.value, bool):
                    sanitized_value = "TRUE" if f.value else "FALSE"
                
                # Thêm vào mệnh đề where
                # Giả định f.field là an toàn (không do người dùng cuối nhập trực tiếp)
                if sanitized_value:
                    where_clauses.append(f"{f.field} {f.operator} {sanitized_value}")

        # Ghép các mệnh đề WHERE (nếu có)
        where_statement = ""
        if where_clauses:
            where_statement = "WHERE " + " AND ".join(where_clauses)

        # Xây dựng truy vấn SOQL động (Query giữ nguyên)
        soql = f"""
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
         {where_statement}
         ORDER BY Contract__r.Created_Date__c DESC
         LIMIT {limit}
         OFFSET {offset}
         """
        
        try:
            query_result = self.sf.query_all(soql)
            records = query_result['records']
            
            if not records:
                return None
            
            # Xử lý DataFrame (Giữ nguyên)
            df = pd.json_normalize(records, sep='.')
            df = df.drop([col for col in df.columns if 'attributes' in col], axis=1)
            
            df.columns = df.columns.str.replace('Contract__r.', 'Contract_', regex=False)
            df.columns = df.columns.str.replace('Product__r.', 'Product_', regex=False)
            df.columns = df.columns.str.replace('Account__r.', 'Account_', regex=False)
            
            return df
            
        except Exception as e:
            # Ném lỗi để endpoint có thể xử lý
            raise Exception(f"Lỗi khi lấy dữ liệu (SOQL có thể sai): {e}")
    
    def transform_data(self, df):
        """Chuyển đổi dữ liệu (Giữ nguyên)"""
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
        
        df_export = df_export[
            (df_export['YEAR'] >= 2015) & 
            (df_export['YEAR'] <= datetime.now().year)
        ]
        
        df_export = df_export.dropna(subset=['Account Name: Account Code'])
        
        df_export = df_export.sort_values(
            by=['Account Name: Account Code', 'YEAR'],
            ascending=[True, False]
        )
        
        df_export = df_export.reset_index(drop=True)
        
        return df_export


# =================================================================
# CẬP NHẬT CÁC ENDPOINTS
# =================================================================

@app.get("/")
async def root():
    """Health check (Cập nhật để hiển thị endpoint mới)"""
    return {
        "status": "ok",
        "message": "Salesforce Contract Products API (Hỗ trợ phân trang và lọc động)",
        "endpoints": {
            "all_products (GET)": "/api/contract-products?limit=100&offset=0",
            "by_account (GET)": "/api/contract-products/by-account?account_code=XXX&limit=100&offset=0",
            "dynamic_filter (POST)": "/api/contract-products/filter"
        }
    }


# THAY ĐỔI: Endpoint này giờ gọi fetch_data mà không có bộ lọc
@app.get("/api/contract-products")
async def get_all_contract_product_details(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0)
):
    """
    Lấy dữ liệu chi tiết sản phẩm hợp đồng đã được xử lý (THEO TRANG).
    """
    try:
        exporter = SalesforceExporter(**SALESFORCE_CONFIG)
        exporter.connect()
        
        # THAY ĐỔI: Gọi fetch_data với filters=None
        df_raw = exporter.fetch_data(limit=limit, offset=offset, filters=None)
        
        if df_raw is None or len(df_raw) == 0:
            return {
                "success": True,
                "count": 0, "limit": limit, "offset": offset,
                "data": [],
                "message": "No contract products found for this page"
            }
        
        df_export = exporter.transform_data(df_raw)
        
        if df_export is None or len(df_export) == 0:
            return {
                "success": True,
                "count": 0, "limit": limit, "offset": offset,
                "data": [],
                "message": "No contract products found after transformation for this page"
            }
        
        df_json = df_export.replace({np.nan: None})
        records = df_json.to_dict(orient='records')
        
        return {
            "success": True,
            "count": len(records),
            "limit": limit,
            "offset": offset,
            "data": records
        }
        
    except HTTPException as http_e:
        return {"success": False, "error": http_e.detail, "status_code": http_e.status_code}
    except Exception as e:
        return {"success": False, "error": f"Error processing data: {str(e)}"}


# THAY ĐỔI: Endpoint này giờ tạo một đối tượng FilterCondition
@app.get("/api/contract-products/by-account")
async def get_contract_details_by_account(
    account_code: str = Query(..., description="Account code to filter by"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0)
):
    """
    Lấy dữ liệu chi tiết sản phẩm hợp đồng (THEO TRANG), lọc theo Account Code.
    """
    try:
        if not account_code or not account_code.strip():
            return {"success": False, "error": "Account code cannot be empty"}
        
        exporter = SalesforceExporter(**SALESFORCE_CONFIG)
        exporter.connect()
        
        # THAY ĐỔI: Tạo một bộ lọc động
        account_filter = [
            FilterCondition(
                field="Contract__r.Account__r.Account_Code__c",
                operator="=",
                value=account_code
            )
        ]
        
        df_raw = exporter.fetch_data(
            limit=limit, 
            offset=offset, 
            filters=account_filter  # Truyền bộ lọc vào
        )
        
        # Logic xử lý kết quả (Giữ nguyên)
        if df_raw is None or len(df_raw) == 0:
            return {
                "success": True,
                "count": 0, "limit": limit, "offset": offset,
                "data": [], "account_code": account_code,
                "message": f"No data found for account code '{account_code}' on this page"
            }
        
        df_export = exporter.transform_data(df_raw)
        
        if df_export is None or len(df_export) == 0:
            return {
                "success": True,
                "count": 0, "limit": limit, "offset": offset,
                "data": [], "account_code": account_code,
                "message": f"No data found (after transform) for account code '{account_code}' on this page"
            }
        
        df_json = df_export.replace({np.nan: None})
        records = df_json.to_dict(orient='records')
        
        return {
            "success": True,
            "count": len(records),
            "limit": limit,
            "offset": offset,
            "data": records,
            "account_code": account_code
        }

    except HTTPException as http_e:
        return {"success": False, "error": http_e.detail, "status_code": http_e.status_code}
    except Exception as e:
        return {"success": False, "error": f"Error processing data for account {account_code}: {str(e)}"}


# =================================================================
# MỚI: Endpoint Lọc Động (Tool mới)
# =================================================================

@app.post("/api/contract-products/filter")
async def get_contract_products_by_dynamic_filter(request: FilterRequest):
    """
    Lấy dữ liệu chi tiết sản phẩm hợp đồng (THEO TRANG), 
    lọc theo NHIỀU điều kiện động được gửi trong body.
    """
    try:
        exporter = SalesforceExporter(**SALESFORCE_CONFIG)
        exporter.connect()
        
        # Gọi fetch_data với các bộ lọc, limit, offset từ body
        df_raw = exporter.fetch_data(
            limit=request.limit, 
            offset=request.offset, 
            filters=request.filters
        )
        
        if df_raw is None or len(df_raw) == 0:
            return {
                "success": True,
                "count": 0, "limit": request.limit, "offset": request.offset,
                "data": [],
                "message": "No contract products found for the specified filters on this page",
                "filters_applied": request.filters
            }
        
        df_export = exporter.transform_data(df_raw)
        
        if df_export is None or len(df_export) == 0:
            return {
                "success": True,
                "count": 0, "limit": request.limit, "offset": request.offset,
                "data": [],
                "message": "No contract products found after transformation for this page",
                "filters_applied": request.filters
            }
        
        df_json = df_export.replace({np.nan: None})
        records = df_json.to_dict(orient='records')
        
        return {
            "success": True,
            "count": len(records),
            "limit": request.limit,
            "offset": request.offset,
            "data": records,
            "filters_applied": request.filters
        }
        
    except HTTPException as http_e:
        return {"success": False, "error": http_e.detail, "status_code": http_e.status_code}
    except Exception as e:
        return {"success": False, "error": f"Error processing data with filters: {str(e)}"}


# =================================================================

if __name__ == "__main__":
    import uvicorn
    # Bạn cần set các biến môi trường (environment variables)
    # SALESFORCE_USERNAME, SALESFORCE_PASSWORD, SALESFORCE_SECURITY_TOKEN
    # trước khi chạy
    uvicorn.run(app, host="0.0.0.0", port=8000)
