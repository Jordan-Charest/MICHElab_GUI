# MICHElab_GUI

  

## Packages pré-requis

  

Les prérequis sont indiqués dans *requirements.txt* et peuvent être installés avec la commande suivante à partir du dossier parent (celui où se trouve *requirements.txt*):

```pip install -r requirements.txt```

## Structure

Le repo a la structure suivante:

```
├── batch_code
│   ├── batch_processing_instructions
│   │   ├── batch_reference
│   │   └── batch_test.txt
│   ├── batch_processing.py
│   └── generate_HDF5_dataset.py
├── gui_tabs
│   ├── tab1.py
│   ├── tab2.py
├── help
│   ├── images
│   │   ├── help1.png
│   │   ├── help2.png
│   └── text
│   │   ├── help1.txt
│       └── help2.txt
├── operations
│   ├── FIJI_macros
│   │   ├── macro_launch.ijm
│   │   └── ROI.ijm
│   ├── operation1.py
│   ├── operation2.py
├── user_config
│   └── readme.txt
├── utils
│   ├── __init__.py
│   ├── module1.py
│   ├── module2.py
├── .gitattributes
├── .gitignore
├── launcher.py
├── README.md
└── requirements.txt
```


