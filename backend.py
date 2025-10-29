"""
FastAPI Backend - Trích xuất dữ liệu từ Salesforce
"""

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from simple_salesforce import Salesforce
from datetime import datetime
import warnings
import os
import numpy as np

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

# Cấu hình Salesforce từ environment variables
SALESFORCE_CONFIG = {
    'username': os.getenv('SALESFORCE_USERNAME'),
    'password': os.getenv('SALESFORCE_PASSWORD'),
    'security_token': os.getenv('SALESFORCE_SECURITY_TOKEN')
}


class SalesforceExporter:
    """Lớp kết nối và xuất dữ liệu từ Salesforce"""
    
    def __init__(self, username, password, security_token):
        self.username = username
        self.password = password
        self.security_token = security_token
        self.sf = None
        
    def connect(self):
        """Kết nối tới Salesforce"""
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
        """Lấy dữ liệu từ Salesforce"""
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
            
            if not records:
                return None
            
            df = pd.json_normalize(records, sep='.')
            df = df.drop([col for col in df.columns if 'attributes' in col], axis=1)
            
            df.columns = df.columns.str.replace('Contract__r.', 'Contract_', regex=False)
            df.columns = df.columns.str.replace('Product__r.', 'Product_', regex=False)
            df.columns = df.columns.str.replace('Account__r.', 'Account_', regex=False)
            
            return df
            
        except Exception as e:
            raise Exception(f"Lỗi khi lấy dữ liệu: {e}")
    
    def transform_data(self, df):
        """Chuyển đổi dữ liệu"""
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


@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "ok",
        "message": "Salesforce Contract Products API",
        "endpoints": {
            "all_products": "/api/contract-products",
            "by_account": "/api/contract-products/by-account?account_code=XXX"
        }
    }


@app.get("/api/contract-products")
async def get_all_contract_product_details():
    """
    Lấy toàn bộ dữ liệu chi tiết sản phẩm hợp đồng đã được xử lý.
    Fetch, transform, and return all contract product details as JSON.
    
    Returns:
        JSON với format: {"success": bool, "count": int, "data": list, "message": str (optional)}
    """
    try:
        # Khởi tạo và kết nối
        exporter = SalesforceExporter(**SALESFORCE_CONFIG)
        exporter.connect()
        
        # Lấy và xử lý dữ liệu
        df_raw = exporter.fetch_data()
        
        if df_raw is None or len(df_raw) == 0:
            return {
                "success": True,
                "count": 0,
                "data": [],
                "message": "No contract products found"
            }
        
        df_export = exporter.transform_data(df_raw)
        
        if df_export is None or len(df_export) == 0:
            return {
                "success": True,
                "count": 0,
                "data": [],
                "message": "No contract products found after transformation"
            }
        
        # Thay thế NaN bằng None để tương thích JSON
        df_json = df_export.replace({np.nan: None})
        records = df_json.to_dict(orient='records')
        
        return {
            "success": True,
            "count": len(records),
            "data": records
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error processing data: {str(e)}"
        }


@app.get("/api/contract-products/by-account")
async def get_contract_details_by_account(
    account_code: str = Query(..., description="Account code to filter by (case-insensitive)")
):
    """
    Lấy dữ liệu chi tiết sản phẩm hợp đồng đã xử lý, lọc theo Account Code.
    Fetch processed contract product details, filtered by a specific Account Code.
    
    Args:
        account_code: Account code để lọc (không phân biệt chữ hoa/thường)
        
    Returns:
        JSON với format: {"success": bool, "count": int, "data": list, "account_code": str, "message": str (optional)}
    """
    try:
        # Validate input
        if not account_code or not account_code.strip():
            return {
                "success": False,
                "error": "Account code cannot be empty"
            }
        
        # Khởi tạo và kết nối
        exporter = SalesforceExporter(**SALESFORCE_CONFIG)
        exporter.connect()
        
        # Lấy và xử lý dữ liệu
        df_raw = exporter.fetch_data()
        
        if df_raw is None or len(df_raw) == 0:
            return {
                "success": True,
                "count": 0,
                "data": [],
                "message": f"No data found (empty source) for account code {account_code}"
            }
        
        df_export = exporter.transform_data(df_raw)
        
        if df_export is None or len(df_export) == 0:
            return {
                "success": True,
                "count": 0,
                "data": [],
                "message": f"No data found (empty after transformation) for account code {account_code}"
            }
        
        # Lọc DataFrame (so sánh không phân biệt chữ hoa/thường và khoảng trắng)
        account_code_clean = account_code.strip().lower()
        filtered_df = df_export[
            df_export['Account Name: Account Code'].astype(str).str.strip().str.lower() == account_code_clean
        ]
        
        if filtered_df.empty:
            return {
                "success": True,
                "count": 0,
                "data": [],
                "message": f"No data found for account code '{account_code}'"
            }
        
        # Thay thế NaN bằng None để tương thích JSON
        df_json = filtered_df.replace({np.nan: None})
        records = df_json.to_dict(orient='records')
        
        return {
            "success": True,
            "count": len(records),
            "data": records,
            "account_code": account_code
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error processing data for account {account_code}: {str(e)}"
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
