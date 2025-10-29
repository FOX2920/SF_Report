"""
FastAPI Backend - Trích xuất dữ liệu từ Salesforce
Đã cập nhật để hỗ trợ Phân Trang Thông Minh (Smart Pagination) 
cho Custom GPT bằng cách trả về `total_records`.
"""

from fastapi import FastAPI, Query, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from simple_salesforce import Salesforce
from datetime import datetime
import warnings
import os
import numpy as np

# Thêm Pydantic và các kiểu dữ liệu
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
# Định nghĩa Model cho Request Lọc Động (Giữ nguyên)
# =================================================================

class FilterCondition(BaseModel):
    """Định nghĩa một điều kiện lọc đơn lẻ"""
    field: str  # Tên trường SOQL, ví dụ: 'Segment__c' hoặc 'Product__r.Family'
    operator: Literal["=", "!=", ">", "<", ">=", "<=", "LIKE"] = "="
    value: str | int | float | bool | None # Giá trị để so sánh

# Bỏ class FilterRequest vì không còn endpoint POST nào sử dụng nó
# class FilterRequest(BaseModel):
#     """
#     Định nghĩa body cho request lọc.
#     Sử dụng Body(...) thay vì Query(...) cho các tham số POST.
#     """
#     filters: List[FilterCondition] = []
#     limit: int = Body(100, ge=1, le=500)
#     offset: int = Body(0, ge=0)


# =================================================================
# CẬP NHẬT Lớp SalesforceExporter
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

    # =================================================================
    # MỚI: Helper Function để tránh lặp code xây dựng WHERE
    # =================================================================
    def _build_where_clause(self, filters: Optional[List[FilterCondition]] = None) -> str:
        """Xây dựng mệnh đề WHERE từ danh sách filter. Trả về string (có thể rỗng)"""
        where_clauses = []
        
        if filters:
            for f in filters:
                if f.operator not in ["=", "!=", ">", "<", ">=", "<=", "LIKE"]:
                    raise ValueError(f"Toán tử không hợp lệ: {f.operator}")

                sanitized_value = ""
                if f.value is None:
                    sanitized_value = "NULL"
                elif isinstance(f.value, str):
                    sanitized_value = f"'{f.value.strip().replace("'", "\\'")}'"
                elif isinstance(f.value, (int, float)):
                    sanitized_value = str(f.value)
                elif isinstance(f.value, bool):
                    sanitized_value = "TRUE" if f.value else "FALSE"
                
                if sanitized_value:
                    where_clauses.append(f"{f.field} {f.operator} {sanitized_value}")

        if where_clauses:
            return "WHERE " + " AND ".join(where_clauses)
        
        return "" # Trả về rỗng nếu không có filter

    # =================================================================
    # MỚI: Hàm chỉ để đếm
    # =================================================================
    def get_count_only(self, filters: Optional[List[FilterCondition]] = None):
        """Chỉ chạy truy vấn COUNT() dựa trên bộ lọc."""
        
        # Sử dụng helper
        where_statement = self._build_where_clause(filters)
        
        try:
            # Dùng COUNT(Id) nhanh hơn COUNT()
            count_soql = f"SELECT COUNT(Id) FROM Contract_Product__c {where_statement}"
            # Dùng .query() cho các truy vấn tổng hợp (aggregate)
            count_result = self.sf.query(count_soql) 
            return count_result['totalSize']
        except Exception as e:
            raise Exception(f"Lỗi khi đếm record (COUNT query): {e}")

    # =================================================================
    # CẬP NHẬT: 'fetch_data' giờ dùng _build_where_clause
    # =================================================================
    def fetch_data(self, limit: int = 100, offset: int = 0, filters: Optional[List[FilterCondition]] = None):
        """
        Lấy dữ liệu từ Salesforce VÀ ĐẾM tổng số record.
        Trả về: (DataFrame, total_records)
        """
        
        # Dùng helper mới
        where_statement = self._build_where_clause(filters)

        # ================================================
        # BƯỚC 1: Truy vấn COUNT()
        # ================================================
        total_records = 0
        try:
            # Gọi hàm count đã có sẵn trong class
            total_records = self.get_count_only(filters)
            
            # Nếu không có record nào, trả về luôn để tiết kiệm
            if total_records == 0:
                return None, 0
                
        except Exception as e:
            # Ném lỗi nếu hàm count thất bại
            raise e
        # ================================================

        # BƯỚC 2: Xây dựng truy vấn SOQL động để lấy dữ liệu
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
                # Không có data TRANG NÀY, nhưng vẫn trả về tổng số
                return None, total_records
            
            # Xử lý DataFrame (Giữ nguyên)
            df = pd.json_normalize(records, sep='.')
            df = df.drop([col for col in df.columns if 'attributes' in col], axis=1)
            
            df.columns = df.columns.str.replace('Contract__r.', 'Contract_', regex=False)
            df.columns = df.columns.str.replace('Product__r.', 'Product_', regex=False)
            df.columns = df.columns.str.replace('Account__r.', 'Account_', regex=False)
            
            return df, total_records # <-- Trả về cả hai
            
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
        "message": "Salesforce Contract Products API (Hỗ trợ phân trang thông minh)",
        "endpoints": {
            "all_products (GET)": "/api/contract-products?limit=100&offset=0",
            "by_account (GET)": "/api/contract-products/by-account?account_code=XXX&limit=100&offset=0",
            "count_all (GET)": "/api/contract-products/count", # <-- MỚI
            "count_by_account (GET)": "/api/contract-products/count/by-account?account_code=XXX" # <-- MỚI
        }
    }


# CẬP NHẬT: Endpoint này giờ trả về metadata
@app.get("/api/contract-products")
async def get_all_contract_product_details(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0)
):
    """
    Lấy dữ liệu chi tiết sản phẩm hợp đồng (THEO TRANG).
    Trả về metadata phân trang.
    """
    try:
        exporter = SalesforceExporter(**SALESFORCE_CONFIG)
        exporter.connect()
        
        df_raw, total_records = exporter.fetch_data(limit=limit, offset=offset, filters=None)
        
        metadata = {
            "total_records": total_records,
            "limit": limit,
            "offset": offset,
            "returned_records": 0
        }
        
        if df_raw is None or len(df_raw) == 0:
            return {
                "success": True,
                "metadata": metadata, 
                "data": [],
                "message": "No contract products found for this page"
            }
        
        df_export = exporter.transform_data(df_raw)
        
        if df_export is None or len(df_export) == 0:
            return {
                "success": True,
                "metadata": metadata, 
                "data": [],
                "message": "No contract products found after transformation for this page"
            }
        
        df_json = df_export.replace({np.nan: None})
        records = df_json.to_dict(orient='records')
        
        metadata["returned_records"] = len(records) 
        
        return {
            "success": True,
            "metadata": metadata, 
            "data": records
        }
        
    except HTTPException as http_e:
        return {"success": False, "error": http_e.detail, "status_code": http_e.status_code}
    except Exception as e:
        return {"success": False, "error": f"Error processing data: {str(e)}"}


# CẬP NHẬT: Endpoint này giờ trả về metadata
@app.get("/api/contract-products/by-account")
async def get_contract_details_by_account(
    account_code: str = Query(..., description="Account code to filter by"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0)
):
    """
    Lấy dữ liệu chi tiết sản phẩm hợp đồng (THEO TRANG), lọc theo Account Code.
    Trả về metadata phân trang.
    """
    try:
        if not account_code or not account_code.strip():
            return {"success": False, "error": "Account code cannot be empty"}
        
        exporter = SalesforceExporter(**SALESFORCE_CONFIG)
        exporter.connect()
        
        account_filter = [
            FilterCondition(
                field="Contract__r.Account__r.Account_Code__c",
                operator="=",
                value=account_code
            )
        ]
        
        df_raw, total_records = exporter.fetch_data(
            limit=limit, 
            offset=offset, 
            filters=account_filter 
        )
        
        metadata = {
            "total_records": total_records,
            "limit": limit,
            "offset": offset,
            "returned_records": 0
        }
        
        if df_raw is None or len(df_raw) == 0:
            return {
                "success": True,
                "metadata": metadata,
                "data": [], "account_code": account_code,
                "message": f"No data found for account code '{account_code}' on this page"
            }
        
        df_export = exporter.transform_data(df_raw)
        
        if df_export is None or len(df_export) == 0:
            return {
                "success": True,
                "metadata": metadata,
                "data": [], "account_code": account_code,
                "message": f"No data found (after transform) for account code '{account_code}' on this page"
            }
        
        df_json = df_export.replace({np.nan: None})
        records = df_json.to_dict(orient='records')
        
        metadata["returned_records"] = len(records) 
        
        return {
            "success": True,
            "metadata": metadata,
            "data": records,
            "account_code": account_code
        }

    except HTTPException as http_e:
        return {"success": False, "error": http_e.detail, "status_code": http_e.status_code}
    except Exception as e:
        return {"success": False, "error": f"Error processing data for account {account_code}: {str(e)}"}


# =================================================================
# MỚI: CÁC ENDPOINT CHỈ ĐẾM (GET)
# =================================================================

@app.get("/api/contract-products/count")
async def count_all_contract_products():
    """
    CHỈ ĐẾM tổng số record, không lọc.
    """
    try:
        exporter = SalesforceExporter(**SALESFORCE_CONFIG)
        exporter.connect()
        
        total_records = exporter.get_count_only(filters=None)
        
        return {
            "success": True,
            "total_records": total_records,
            "filters_applied": []
        }

    except HTTPException as http_e:
        return {"success": False, "error": http_e.detail, "status_code": http_e.status_code}
    except Exception as e:
        return {"success": False, "error": f"Error counting data: {str(e)}"}

@app.get("/api/contract-products/count/by-account")
async def count_contract_details_by_account(
    account_code: str = Query(..., description="Account code to filter by"),
):
    """
    CHỈ ĐẾM tổng số record, lọc theo Account Code.
    """
    try:
        if not account_code or not account_code.strip():
            return {"success": False, "error": "Account code cannot be empty"}
        
        exporter = SalesforceExporter(**SALESFORCE_CONFIG)
        exporter.connect()
        
        account_filter = [
            FilterCondition(
                field="Contract__r.Account__r.Account_Code__c",
                operator="=",
                value=account_code
            )
        ]
        
        total_records = exporter.get_count_only(filters=account_filter)
        
        return {
            "success": True,
            "total_records": total_records,
            "filters_applied": account_filter
        }

    except HTTPException as http_e:
        return {"success": False, "error": http_e.detail, "status_code": http_e.status_code}
    except Exception as e:
        return {"success": False, "error": f"Error counting data for account {account_code}: {str(e)}"}


# =================================================================
# ĐÃ XÓA ENDPOINT: /api/contract-products/filter (POST)
# =================================================================


# =================================================================
# ĐÃ XÓA ENDPOINT: /api/contract-products/count (POST)
# =================================================================


# =================================================================

if __name__ == "__main__":
    import uvicorn
    # Bạn cần set các biến môi trường (environment variables)
    # SALESFORCE_USERNAME, SALESFORCE_PASSWORD, SALESFORCE_SECURITY_TOKEN
    # trước khi chạy
    # Ví dụ (chỉ để test, không dùng trong production):
    # os.environ['SALESFORCE_USERNAME'] = 'your_user@example.com'
    # os.environ['SALESFORCE_PASSWORD'] = 'your_password'
    # os.environ['SALESFORCE_SECURITY_TOKEN'] = 'your_token'
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
