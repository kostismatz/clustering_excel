import pandas as pd

data = {
    "Product": [
        "Industrial Pump",
        "Hydraulic Pump",
        "Water Pump",
        "Gaming Laptop",
        "Business Laptop",
        "Office PC",
        "Smartphone Samsung",
        "iPhone 14",
        "Android Phone",
        "Cement Bag",
        "Steel Rod",
        "Construction Materials",
        "Air Conditioner",
        "AC Unit Split",
        "Cooling System"
    ],

    "Category": [
        "Industrial",
        "Industrial",
        "Industrial",
        "IT",
        "IT",
        "IT",
        "Mobile",
        "Mobile",
        "Mobile",
        "Construction",
        "Construction",
        "Construction",
        "Home Appliances",
        "Home Appliances",
        "Home Appliances"
    ],

    "Description": [
        "High pressure industrial pump",
        "Hydraulic system pump",
        "Water circulation pump",
        "High performance gaming laptop",
        "Laptop for business use",
        "Desktop computer for office",
        "Samsung Android smartphone",
        "Apple smartphone device",
        "Android mobile phone device",
        "Construction cement material",
        "Steel reinforcement rod",
        "Materials for building works",
        "Cooling air conditioner system",
        "Split AC unit for rooms",
        "Cooling and ventilation system"
    ],

    "Price": [
        1200, 1350, 1100,
        1500, 1300, 900,
        800, 1200, 600,
        50, 70, 40,
        500, 650, 700
    ]
}

df = pd.DataFrame(data)

df.to_excel("test_data.xlsx", index=False)

print("Excel created: test_data.xlsx")
