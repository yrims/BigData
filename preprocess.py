# pre-processing the excel data

import openpyxl


fn = 'data\HS\G8\G8-AC(7X15)20160608-001_Export.xlsx'
wb = openpyxl.load_workbook(fn)
print(type(wb))

all_sheets = wb.get_sheet_names()
print('all sheets :', all_sheets)

ws = wb.get_active_sheet()
print('ws :', ws)