from PyInstaller.utils.hooks import collect_data_files

hiddenimports = [
    'gradio'
]

# Use collect_data_files to find data files. Replace 'gradio' with the correct package name if it's different.
datas = collect_data_files('gradio')
