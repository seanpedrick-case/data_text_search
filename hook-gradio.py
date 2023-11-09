from PyInstaller.utils.hooks import collect_data_files

hiddenimports = [
    'gradio',
    # Add any other submodules that PyInstaller doesn't detect
]

# Use collect_data_files to find data files. Replace 'gradio' with the correct package name if it's different.
datas = collect_data_files('gradio')
