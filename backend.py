"""
FastAPI Backend - Trích xuất dữ liệu từ Salesforce và xuất Excel
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from simple_salesforce import Salesforce
from datetime import datetime
import warnings
import os
import numpy as np # <-- THÊM DÒNG NÀY
from dotenv import load_dotenv # <-- THÊM DÒNG NÀY

# Load environment variables từ file .env
load_dotenv() # <-- THÊM DÒNG NÀY

warnings.filterwarnings('ignore')

app = FastAPI(title="Salesforce Data Exporter")

# CORS middleware để cho phép React kết nối
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
        
        return df_export
    
    def export_to_excel(self, df, output_file):
        """Xuất dữ liệu ra Excel"""
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df.to_excel(
                writer,
                sheet_name='Chi tết sản phẩm theo KH',
                index=False
            )
            
            worksheet = writer.sheets['Chi tết sản phẩm theo KH']
            
            for idx, col in enumerate(df.columns, 1):
                max_length = max(
                    df[col].astype(str).apply(len).max(),
                    len(str(col))
                ) + 2
                max_length = min(max_length, 50)
                worksheet.column_dimensions[chr(64 + idx)].width = max_length


@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "ok",
        "message": "Salesforce Data Exporter API"
    }


@app.get("/api/export")
async def export_data():
    """
    API endpoint để export dữ liệu từ Salesforce
    """
    try:
        # Khởi tạo exporter
        exporter = SalesforceExporter(**SALESFORCE_CONFIG)
        
        # Kết nối
        exporter.connect()
        
        # Lấy dữ liệu
        df_raw = exporter.fetch_data()
        if df_raw is None or len(df_raw) == 0:
            raise HTTPException(status_code=404, detail="Không có dữ liệu")
        
        # Chuyển đổi
        df_export = exporter.transform_data(df_raw)
        if df_export is None or len(df_export) == 0:
            raise HTTPException(status_code=500, detail="Lỗi chuyển đổi dữ liệu")
        
        # Xuất Excel
        output_file = 'Chi_tiet_san_pham_KH_Active.xlsx'
        exporter.export_to_excel(df_export, output_file)
        
        # Trả về file
        return FileResponse(
            path=output_file,
            filename=output_file,
            media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
async def get_stats():
    """
    API endpoint để lấy thống kê dữ liệu
    """
    try:
        exporter = SalesforceExporter(**SALESFORCE_CONFIG)
        exporter.connect()
        
        df_raw = exporter.fetch_data()
        if df_raw is None:
            return {"total_records": 0}
        
        df_export = exporter.transform_data(df_raw)
        
        return {
            "total_records": len(df_export),
            "total_customers": int(df_export['Account Name: Account Code'].nunique()),
            "year_range": f"{int(df_export['YEAR'].min())} - {int(df_export['YEAR'].max())}",
            "last_updated": datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- API MỚI ĐƯỢC THÊM VÀO ---
@app.get("/api/data")
async def get_data_as_json():
    """
    API endpoint để lấy dữ liệu df_export dưới dạng JSON
    """
    try:
        exporter = SalesforceExporter(**SALESFORCE_CONFIG)
        exporter.connect()
        
        df_raw = exporter.fetch_data()
        if df_raw is None:
             raise HTTPException(status_code=404, detail="Không có dữ liệu")
        
        df_export = exporter.transform_data(df_raw)
        if df_export is None or len(df_export) == 0:
             raise HTTPException(status_code=500, detail="Lỗi chuyển đổi dữ liệu")
        
        # Thay thế NaN (không hợp lệ trong JSON) bằng None (null trong JSON)
        # df_export_json = df_export.where(pd.notnull(df_export), None) # <-- CÁCH CŨ
        
        # FIX: Thay thế NaN (từ float, int, object, v.v.) bằng None (null)
        # .replace() đáng tin cậy hơn .where() cho mục đích này
        df_export_json = df_export.replace({np.nan: None})
        
        # Chuyển đổi DataFrame sang JSON định dạng 'records' (list of dicts)
        data_json = df_export_json.to_dict(orient='records')
        
        return data_json
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# --- KẾT THÚC API MỚI ---


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)