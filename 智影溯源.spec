# -*- mode: python ; coding: utf-8 -*-

added_datas = [
    ("app.py", "."),
    ("game.py", "."),
    ("part1.py", "."),
    ("model.py", "."),
    ("model_config.json", "."),
    ("cases_manifest.json", "."),
    ("requirements.txt", "."),
    ("cases", "cases"),
    ("data", "data"),
    ("loss_fig", "loss_fig"),
    ("model_parameter", "model_parameter"),
]

a = Analysis(
    ['launcher.py'],
    pathex=[],
    binaries=[],
    datas=added_datas,
    hiddenimports=[
        'streamlit', 'streamlit.web.cli',
        'streamlit.runtime', 'streamlit.runtime.scriptrunner',
        'torch', 'torchvision',
        'numpy', 'PIL', 'cv2',
        'matplotlib', 'pydicom',
        'webview', 'pydeck', 'altair', 'tornado',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],  # 不要随便 exclude
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='智影溯源',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='智影溯源',
)
