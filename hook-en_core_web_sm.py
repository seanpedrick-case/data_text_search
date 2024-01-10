from PyInstaller.utils.hooks import collect_data_files

hiddenimports = [
    'en_core_web_sm'
]

# Use collect_data_files to find data files. Replace 'en_core_web_sm' with the correct package name if it's different.
datas = collect_data_files('en_core_web_sm')
