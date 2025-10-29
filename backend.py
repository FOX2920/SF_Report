"""
FastAPI Backend - Trích xuất dữ liệu từ Salesforce
Đã cập nhật để hỗ trợ Phân Trang (Pagination) VÀ các Endpoint Phân tích
"""

from fastapi import FastAPI, Query, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from simple_salesforce import Salesforce
from datetime import datetime
import warnings
import os
import numpy as np
import time
from threading import Lock
from typing import Optional, List, Dict, Any

# --- Cảnh báo & Cài đặt ---
warnings.filterwarnings('ignore')
# Để chạy phân tích Giỏ hàng, bạn cần cài đặt mlxtend:
# pip install mlxtend
try:
    from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import apriori, association_rules
    MLXTEND_AVAILABLE = True
except ImportError:
    MLXTEND_AVAILABLE = False


# --- Khởi tạo FastAPI App ---
app = FastAPI(title="Salesforce Contract Products API & Analytics")

# --- CORS middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Cấu hình Salesforce ---
SALESFORCE_CONFIG = {
    'username': os.getenv('SALESFORCE_USERNAME', 'YOUR_DEFAULT_USERNAME_IF_NEEDED'),
    'password': os.getenv('SALESFORCE_PASSWORD', 'YOUR_DEFAULT_PASSWORD_IF_NEEDED'),
    'security_token': os.getenv('SALESFORCE_SECURITY_TOKEN', 'YOUR_DEFAULT_TOKEN_IF_NEEDED')
}

# --- Bộ đệm (Cache) cho Dữ liệu Phân tích ---
# Các phân tích yêu cầu toàn bộ dữ liệu, không thể phân trang.
# Chúng ta sẽ cache dữ liệu để tránh gọi Salesforce mỗi lần.
analytics_cache = {
    "df": pd.DataFrame(),
    "last_updated": 0
}
CACHE_DURATION = 3600  # 1 giờ (tính bằng giây)
cache_lock = Lock()


# --- LỚP KẾT NỐI SALESFORCE (Nguyên bản) ---
class SalesforceExporter:
    """Lớp kết nối và xuất dữ liệu từ Salesforce"""
    
    def __init__(self, username, password, security_token):
        self.username = username
        self.password = password
        self.security_token = security_token
        self.sf = None
        
        if not all([self.username, self.password, self.security_token]):
            print("Cảnh báo: Thiếu thông tin đăng nhập Salesforce từ biến môi trường.")
            
    def connect(self) -> bool:
        """Kết nối tới Salesforce"""
        if not all([self.username, self.password, self.security_token]):
             raise HTTPException(status_code=500, detail="Thiếu thông tin cấu hình Salesforce (biến môi trường).")
        try:
            self.sf = Salesforce(
                username=self.username,
                password=self.password,
                security_token=self.security_token
            )
            return True
        except Exception as e:
            print(f"Lỗi kết nối Salesforce: {e}")
            raise HTTPException(status_code=502, detail=f"Không thể kết nối tới Salesforce: {e}")

    def get_data_from_salesforce(self, account_code: str, limit: int, offset: int) -> pd.DataFrame:
        """Hàm gốc: Lấy dữ liệu theo từng Account Code với phân trang"""
        soql_query = f"""
            SELECT 
                Account.Name, 
                Account.Account_Code__c, 
                Product__r.Product_SKU__c, 
                Product__r.M_t_s_n_ph_m__c,
                Product__r.STONE_Color_Type__c, 
                Product__r.Product_Family__c,
                Contract__r.Name, 
                Contract__r.CreatedDate,
                Contract__r.Segment__c,
                Name, 
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
                Total_Price_USD__c, 
                YEAR__c
            FROM Contract_Product__c
            WHERE Account.Account_Code__c = :account_code
            ORDER BY Contract__r.CreatedDate DESC
            LIMIT {limit}
            OFFSET {offset}
        """
        try:
            query_result = self.sf.query_all(soql_query, account_code=account_code)
            records = [
                {
                    'Account Name': r['Account']['Name'],
                    'Account Code': r['Account']['Account_Code__c'],
                    'Product: Product SKU': r['Product__r']['Product_SKU__c'] if r['Product__r'] else None,
                    'Product: Mô tả sản phẩm': r['Product__r']['M_t_s_n_ph_m__c'] if r['Product__r'] else None,
                    'Product: STONE Color Type': r['Product__r']['STONE_Color_Type__c'] if r['Product__r'] else None,
                    'Product: Product Family': r['Product__r']['Product_Family__c'] if r['Product__r'] else None,
                    'Contract Name': r['Contract__r']['Name'] if r['Contract__r'] else None,
                    'Created Date (C)': r['Contract__r']['CreatedDate'] if r['Contract__r'] else None,
                    'Segment': r['Contract__r']['Segment__c'] if r['Contract__r'] else None,
                    'Contract Product Name': r['Name'],
                    'Length': r['Length__c'],
                    'Width': r['Width__c'],
                    'Height': r['Height__c'],
                    'Quantity': r['Quantity__c'],
                    'Crates': r['Crates__c'],
                    'm2': r['m2__c'],
                    'm3': r['m3__c'],
                    'Tons': r['Tons__c'],
                    'Cont': r['Cont__c'],
                    'Sales Price': r['Sales_Price__c'],
                    'Charge Unit (PI)': r['Charge_Unit_PI__c'],
                    'Total Price (USD)': r['Total_Price_USD__c'],
                    'YEAR': r['YEAR__c']
                }
                for r in query_result.get('records', [])
            ]
            return pd.DataFrame(records)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Lỗi truy vấn SOQL: {e}")

    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Hàm gốc: Chuyển đổi dữ liệu thô"""
        if df.empty:
            return pd.DataFrame()
            
        def convert_to_gmt_plus_7(date_str):
            if pd.isnull(date_str):
                return None
            try:
                dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                return dt.strftime('%d/%m/%Y')
            except Exception:
                return None

        df['Created Date (C)'] = df['Created Date (C)'].apply(convert_to_gmt_plus_7)
        
        # Lấy năm từ 'YEAR' hoặc 'Created Date (C)'
        def get_year(row):
            if pd.notnull(row['YEAR']) and row['YEAR'] != 0:
                try:
                    return int(row['YEAR'])
                except ValueError:
                    pass
            if pd.notnull(row['Created Date (C)']):
                try:
                    return datetime.strptime(row['Created Date (C)'], '%d/%m/%Y').year
                except Exception:
                    pass
            return None
        
        df['YEAR'] = df.apply(get_year, axis=1)

        # Đổi tên cột
        df = df.rename(columns={
            'Product: Product SKU': 'Product SKU',
            'Product: Mô tả sản phẩm': 'Product Description',
            'Product: STONE Color Type': 'STONE Color Type',
            'Product: Product Family': 'Product Family'
        })

        # Sắp xếp lại các cột
        columns_order = [
            'Account Name', 'Account Code', 'STONE Color Type', 'Product SKU', 
            'Contract Product Name', 'YEAR', 'Product Description', 
            'Length', 'Width', 'Height', 'Quantity', 'Crates', 'm2', 'm3', 'Tons', 
            'Cont', 'Sales Price', 'Charge Unit (PI)', 'Total Price (USD)', 
            'Product Family', 'Segment', 'Contract Name', 'Created Date (C)'
        ]
        
        # Đảm bảo tất cả các cột đều tồn tại
        for col in columns_order:
            if col not in df.columns:
                df[col] = np.nan
                
        df = df[columns_order]
        return df
        
    # --- HÀM MỚI: Lấy TẤT CẢ dữ liệu cho phân tích ---
    def get_all_data_from_salesforce(self) -> pd.DataFrame:
        """
        Lấy TẤT CẢ dữ liệu Contract Product (không phân trang, không lọc) 
        để phục vụ cho cache phân tích.
        """
        soql_query = """
            SELECT 
                Account.Name, 
                Account.Account_Code__c, 
                Product__r.Product_SKU__c, 
                Product__r.M_t_s_n_ph_m__c,
                Product__r.STONE_Color_Type__c, 
                Product__r.Product_Family__c,
                Contract__r.Name, 
                Contract__r.CreatedDate,
                Contract__r.Segment__c,
                Name, 
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
                Total_Price_USD__c, 
                YEAR__c
            FROM Contract_Product__c
            WHERE Account.Account_Code__c != null 
              AND Contract__r.CreatedDate != null
              AND Total_Price_USD__c > 0
            ORDER BY Contract__r.CreatedDate DESC
        """
        try:
            print("Đang truy vấn TẤT CẢ dữ liệu từ Salesforce cho cache...")
            query_result = self.sf.query_all(soql_query)
            records = [
                {
                    'Account Name': r['Account']['Name'],
                    'Account Code': r['Account']['Account_Code__c'],
                    'Product: Product SKU': r['Product__r']['Product_SKU__c'] if r['Product__r'] else None,
                    'Product: Mô tả sản phẩm': r['Product__r']['M_t_s_n_ph_m__c'] if r['Product__r'] else None,
                    'Product: STONE Color Type': r['Product__r']['STONE_Color_Type__c'] if r['Product__r'] else None,
                    'Product: Product Family': r['Product__r']['Product_Family__c'] if r['Product__r'] else None,
                    'Contract Name': r['Contract__r']['Name'] if r['Contract__r'] else None,
                    'Created Date (C)': r['Contract__r']['CreatedDate'] if r['Contract__r'] else None,
                    'Segment': r['Contract__r']['Segment__c'] if r['Contract__r'] else None,
                    'Contract Product Name': r['Name'],
                    'Length': r['Length__c'],
                    'Width': r['Width__c'],
                    'Height': r['Height__c'],
                    'Quantity': r['Quantity__c'],
                    'Crates': r['Crates__c'],
                    'm2': r['m2__c'],
                    'm3': r['m3__c'],
                    'Tons': r['Tons__c'],
                    'Cont': r['Cont__c'],
                    'Sales Price': r['Sales_Price__c'],
                    'Charge Unit (PI)': r['Charge_Unit_PI__c'],
                    'Total Price (USD)': r['Total_Price_USD__c'],
                    'YEAR': r['YEAR__c']
                }
                for r in query_result.get('records', [])
            ]
            print(f"Đã truy vấn xong, có {len(records)} dòng thô.")
            return pd.DataFrame(records)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Lỗi truy vấn SOQL (lấy tất cả): {e}")

# --- HÀM HELPER: Quản lý Cache Phân tích ---

def get_exporter() -> SalesforceExporter:
    """Hàm Dependency Injection cho FastAPI để tạo Exporter"""
    return SalesforceExporter(
        username=SALESFORCE_CONFIG['username'],
        password=SALESFORCE_CONFIG['password'],
        security_token=SALESFORCE_CONFIG['security_token']
    )

def get_cached_analytics_data(exporter: SalesforceExporter = Depends(get_exporter)) -> pd.DataFrame:
    """
    Hàm helper (sử dụng Depends) để lấy dữ liệu từ cache.
    Nếu cache cũ hoặc trống, nó sẽ tự động kết nối Salesforce và làm mới.
    """
    with cache_lock:
        now = time.time()
        if (now - analytics_cache["last_updated"]) > CACHE_DURATION or analytics_cache["df"].empty:
            print("Cache phân tích bị rỗng hoặc đã cũ. Đang làm mới...")
            try:
                if not exporter.connect():
                    raise HTTPException(status_code=503, detail="Không thể kết nối Salesforce để làm mới cache")
                
                df_raw = exporter.get_all_data_from_salesforce()
                if df_raw is None or df_raw.empty:
                    raise HTTPException(status_code=404, detail="Không tìm thấy dữ liệu (thô) cho phân tích")
                
                df_export = exporter.transform_data(df_raw)
                if df_export is None or df_export.empty:
                    raise HTTPException(status_code=404, detail="Không tìm thấy dữ liệu (đã chuyển đổi) cho phân tích")
                
                # Chuyển đổi cột ngày tháng để phân tích
                df_export['Created Date (C)'] = pd.to_datetime(df_export['Created Date (C)'], dayfirst=True, errors='coerce')
                df_export['Total Price (USD)'] = pd.to_numeric(df_export['Total Price (USD)'], errors='coerce').fillna(0)

                analytics_cache["df"] = df_export
                analytics_cache["last_updated"] = now
                print("Làm mới cache thành công.")
            
            except HTTPException as http_e:
                # Ném lại lỗi HTTPException
                raise http_e
            except Exception as e:
                # Bắt các lỗi khác
                raise HTTPException(status_code=500, detail=f"Lỗi khi làm mới cache: {e}")
        
        return analytics_cache["df"]

# --- ENDPOINT GỐC (Nguyên bản) ---
@app.get("/contract-products-by-account")
def get_contract_products_by_account(
    account_code: str = Query(..., description="Mã khách hàng (ví dụ: X09)"),
    limit: int = Query(100, ge=1, le=1000, description="Số lượng dòng tối đa"),
    offset: int = Query(0, ge=0, description="Vị trí bắt đầu (dùng cho phân trang)")
):
    """
    API GỐC: Lấy danh sách sản phẩm theo hợp đồng của một khách hàng, có phân trang.
    """
    try:
        exporter = get_exporter() # Tạo instance mới cho mỗi lần gọi
        if not exporter.connect():
            return {"success": False, "error": "Không thể kết nối tới Salesforce"}

        df_raw = exporter.get_data_from_salesforce(account_code, limit, offset)
        
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


# --- CÁC ENDPOINT PHÂN TÍCH MỚI ---

@app.get("/analytics/cache-status")
def get_cache_status():
    """API tiện ích: Kiểm tra trạng thái của cache phân tích."""
    with cache_lock:
        last_update_time = "Chưa có dữ liệu"
        if analytics_cache["last_updated"] > 0:
            last_update_time = datetime.fromtimestamp(analytics_cache["last_updated"]).strftime('%Y-%m-%d %H:%M:%S')
        
        return {
            "success": True,
            "cache_populated": not analytics_cache["df"].empty,
            "row_count": len(analytics_cache["df"]),
            "last_updated_timestamp": analytics_cache["last_updated"],
            "last_updated_human": last_update_time,
            "expires_in_seconds": max(0, CACHE_DURATION - (time.time() - analytics_cache["last_updated"]))
        }

@app.get("/analytics/refresh-cache")
def force_refresh_cache(df: pd.DataFrame = Depends(get_cached_analytics_data)):
    """
    API tiện ích: Bắt buộc làm mới cache (bằng cách xóa và gọi lại).
    LƯU Ý: Endpoint này sẽ xóa cache cũ và gọi get_cached_analytics_data
    để nạp lại dữ liệu mới.
    """
    with cache_lock:
        analytics_cache["df"] = pd.DataFrame()
        analytics_cache["last_updated"] = 0
    
    # Gọi hàm phụ thuộc để nạp lại cache
    try:
        get_cached_analytics_data(get_exporter())
        return {"success": True, "message": "Cache đã được làm mới."}
    except HTTPException as e:
        return {"success": False, "error": f"Lỗi khi làm mới cache: {e.detail}"}


@app.get("/analytics/rfm-segmentation")
def get_rfm_segmentation(df: pd.DataFrame = Depends(get_cached_analytics_data)):
    """
    Bài toán 1: Phân khúc khách hàng (RFM).
    Trả về điểm RFM và phân khúc cho TẤT CẢ khách hàng.
    """
    try:
        if df.empty:
            raise HTTPException(status_code=404, detail="Dữ liệu cache rỗng.")
            
        df_rfm = df.copy()
        df_rfm = df_rfm.dropna(subset=['Created Date (C)'])
        
        # 1. Tính toán R-F-M
        snapshot_date = datetime.now()
        
        rfm_data = df_rfm.groupby('Account Name').agg(
            Recency=('Created Date (C)', lambda x: (snapshot_date - x.max()).days),
            Frequency=('Contract Name', 'nunique'),
            Monetary=('Total Price (USD)', 'sum')
        ).reset_index()

        # 2. Chấm điểm R, F, M (chia thành 4 khoảng)
        # Điểm R: Càng nhỏ càng tốt (gần đây)
        rfm_data['R_Score'] = pd.qcut(rfm_data['Recency'], 4, labels=[4, 3, 2, 1], duplicates='drop').astype(int)
        # Điểm F & M: Càng lớn càng tốt
        rfm_data['F_Score'] = pd.qcut(rfm_data['Frequency'].rank(method='first'), 4, labels=[1, 2, 3, 4], duplicates='drop').astype(int)
        rfm_data['M_Score'] = pd.qcut(rfm_data['Monetary'].rank(method='first'), 4, labels=[1, 2, 3, 4], duplicates='drop').astype(int)

        # 3. Tạo điểm tổng hợp và Phân khúc
        rfm_data['RFM_Score'] = rfm_data['R_Score'].astype(str) + rfm_data['F_Score'].astype(str) + rfm_data['M_Score'].astype(str)
        
        # Định nghĩa phân khúc (ví dụ)
        segment_map = {
            r'[3-4][3-4][3-4]': 'Champions',
            r'[1-2][3-4][3-4]': 'Loyal Customers',
            r'[3-4][1-2][3-4]': 'Potential Loyalists',
            r'[3-4][3-4][1-2]': 'Recent Customers',
            r'[1-2][1-2][3-4]': 'Customers Needing Attention',
            r'[1-2][3-4][1-2]': 'At Risk',
            r'[1-2][1-2][1-2]': 'Lost'
        }
        
        rfm_data['Segment'] = rfm_data['RFM_Score'].replace(segment_map, regex=True)
        # Bất kỳ ai không khớp sẽ là 'Others'
        rfm_data['Segment'] = rfm_data['Segment'].apply(lambda x: x if x in segment_map.values() else 'Others')

        records = rfm_data.to_dict(orient='records')
        return {"success": True, "count": len(records), "data": records}

    except Exception as e:
        return {"success": False, "error": f"Lỗi phân tích RFM: {str(e)}"}

@app.get("/analytics/segment-summary")
def get_segment_summary(df: pd.DataFrame = Depends(get_cached_analytics_data)):
    """
    Bài toán 1 (Phụ): Lấy tóm tắt số lượng khách hàng theo từng phân khúc.
    """
    try:
        # Tái sử dụng logic từ endpoint RFM
        rfm_result = get_rfm_segmentation(df)
        if not rfm_result.get("success"):
            return rfm_result # Trả về lỗi nếu có

        rfm_df = pd.DataFrame(rfm_result.get("data"))
        if rfm_df.empty:
            return {"success": True, "data": {}}
            
        segment_counts = rfm_df['Segment'].value_counts().to_dict()
        return {"success": True, "data": segment_counts}
        
    except Exception as e:
        return {"success": False, "error": f"Lỗi tóm tắt phân khúc: {str(e)}"}

@app.get("/analytics/market-basket")
def get_market_basket_analysis(
    min_confidence: float = Query(0.3, ge=0.1, le=1.0),
    df: pd.DataFrame = Depends(get_cached_analytics_data)
):
    """
    Bài toán 2: Phân tích giỏ hàng (Luật kết hợp).
    Tìm các sản phẩm (theo Product Family) thường được mua cùng nhau.
    """
    if not MLXTEND_AVAILABLE:
        raise HTTPException(status_code=501, detail="Thư viện 'mlxtend' chưa được cài đặt. Không thể thực hiện phân tích giỏ hàng.")
        
    try:
        df_basket = df.dropna(subset=['Contract Name', 'Product Family'])
        
        # 1. Chuẩn bị dữ liệu giao dịch
        transactions = df_basket.groupby('Contract Name')['Product Family'].apply(list).values.tolist()
        
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        basket_df = pd.DataFrame(te_ary, columns=te.columns_)
        
        # 2. Chạy Apriori
        frequent_itemsets = apriori(basket_df, min_support=0.01, use_colnames=True)
        if frequent_itemsets.empty:
            return {"success": True, "message": "Không tìm thấy bộ sản phẩm phổ biến nào.", "data": []}

        # 3. Tạo luật kết hợp
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        
        if rules.empty:
            return {"success": True, "message": f"Không tìm thấy luật kết hợp nào với confidence > {min_confidence}", "data": []}
            
        # 4. Lọc và định dạng kết quả
        rules_filtered = rules[rules['lift'] > 1.0]
        rules_filtered['antecedents'] = rules_filtered['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules_filtered['consequents'] = rules_filtered['consequents'].apply(lambda x: ', '.join(list(x)))
        
        rules_filtered = rules_filtered.sort_values(by='lift', ascending=False)
        
        records = rules_filtered[['antecedents', 'consequents', 'support', 'confidence', 'lift']].to_dict(orient='records')
        return {"success": True, "count": len(records), "data": records}

    except Exception as e:
        return {"success": False, "error": f"Lỗi phân tích giỏ hàng: {str(e)}"}

@app.get("/analytics/monthly-sales")
def get_monthly_sales_history(df: pd.DataFrame = Depends(get_cached_analytics_data)):
    """
    Bài toán 3: Dữ liệu cho Dự báo Doanh thu.
    Trả về tổng doanh thu (Total Price USD) theo từng tháng.
    """
    try:
        df_sales = df.dropna(subset=['Created Date (C)', 'Total Price (USD)'])
        
        # Resample: 'ME' = Month End
        monthly_sales = df_sales.set_index('Created Date (C)').resample('ME')['Total Price (USD)'].sum().reset_index()
        
        monthly_sales['Created Date (C)'] = monthly_sales['Created Date (C)'].dt.strftime('%Y-%m-%d')
        monthly_sales = monthly_sales.rename(columns={'Created Date (C)': 'month', 'Total Price (USD)': 'total_sales_usd'})
        
        records = monthly_sales.to_dict(orient='records')
        return {"success": True, "count": len(records), "data": records}

    except Exception as e:
        return {"success": False, "error": f"Lỗi tổng hợp doanh thu: {str(e)}"}

@app.get("/analytics/product-performance")
def get_product_performance(
    group_by: str = Query("Product Family", enum=["Product Family", "Segment", "STONE Color Type"]),
    df: pd.DataFrame = Depends(get_cached_analytics_data)
):
    """
    Bài toán 4: Phân tích Hiệu suất (Pareto 80/20).
    Phân tích doanh thu theo 'Product Family', 'Segment', hoặc 'STONE Color Type'.
    """
    try:
        if group_by not in df.columns:
            raise HTTPException(status_code=400, detail=f"Cột '{group_by}' không hợp lệ.")

        df_perf = df.dropna(subset=[group_by, 'Total Price (USD)'])

        # 1. Tổng hợp doanh thu
        perf_data = df_perf.groupby(group_by)['Total Price (USD)'].sum().reset_index()
        perf_data = perf_data.sort_values(by='Total Price (USD)', ascending=False)
        
        # 2. Tính Pareto (Cumulative Percentage)
        perf_data['cumulative_sales'] = perf_data['Total Price (USD)'].cumsum()
        perf_data['total_sales'] = perf_data['Total Price (USD)'].sum()
        perf_data['cumulative_percent'] = (perf_data['cumulative_sales'] / perf_data['total_sales'])
        
        # Gắn thẻ Pareto
        perf_data['pareto_group'] = np.where(perf_data['cumulative_percent'] <= 0.8, 'Top 80% Revenue', 'Bottom 20% Revenue')
        
        records = perf_data.to_dict(orient='records')
        return {"success": True, "count": len(records), "data": records}

    except Exception as e:
        return {"success": False, "error": f"Lỗi phân tích hiệu suất: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    # Lấy cổng từ biến môi trường, nếu không có thì dùng 8000
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
