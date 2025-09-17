# 📊 Data Structure Documentation

This document describes the expected data format for the Berlin Project Sales Analysis System.

## CSV File Format

The system expects CSV files with the following column structure:

### Required Columns

| Column Name | Data Type | Description | Example Values |
|-------------|-----------|-------------|----------------|
| `Menu Section` | String | Category section of the menu | "Beer", "Signature cocktails", "Happy Hour" |
| `Menu Item` | String | Name of the product/item | "Craft IPA", "Moscow Mule", "Margarita" |
| `Size` | String | Size or portion description | "12oz", "Large", "Small", "Regular" |
| `Portion` | String | Number of portions | "1", "2", "Half" |
| `Category` | String | Product category | "Beverage", "Food", "Appetizer" |
| `Unit Price` | String | Price per unit (with $ symbol) | "$5.50", "$12.00", "$8.75" |
| `Quantity` | Integer | Number of units sold | 25, 10, 3 |
| `Sales` | String | Total revenue (with $ symbol) | "$137.50", "$120.00", "$26.25" |
| `% of Sales` | String | Percentage of total sales (with % symbol) | "15.2%", "8.5%", "2.1%" |

### Sample Data Structure

```csv
Menu Section,Menu Item,Size,Portion,Category,Unit Price,Quantity,Sales,% of Sales
Beer,Craft IPA,12oz,1,Beverage,$5.50,25,$137.50,15.2%
Signature cocktails,Moscow Mule,Regular,1,Beverage,$12.00,10,$120.00,8.5%
Happy Hour,Margarita,Regular,1,Beverage,$8.75,3,$26.25,2.1%
```

## Data Processing Notes

### Automatic Cleaning
The system automatically processes the following:
- Removes `$` symbols from price columns
- Removes `%` symbols from percentage columns
- Converts string numbers to appropriate data types
- Filters out zero-quantity items

### Data Validation
The system validates:
- Required columns are present
- Numeric columns contain valid numbers
- No missing critical data

### Supported Formats
- CSV files with comma separators
- UTF-8 encoding
- First row can be a title (automatically skipped)

## Creating Test Data

To test the system without using confidential data:

1. Create a CSV file with the structure above
2. Use sample product names and categories
3. Generate realistic sales data
4. Ensure all required columns are present

## Security Considerations

- Never commit real sales data to version control
- Use the `.gitignore` file to exclude CSV files
- Store confidential data in secure, local locations only
- Consider data anonymization for testing purposes

## Troubleshooting

### Common Issues
- **Missing columns**: Ensure all required columns are present
- **Invalid price format**: Use `$X.XX` format for prices
- **Invalid percentage format**: Use `X.X%` format for percentages
- **Empty rows**: Remove any completely empty rows

### Data Quality Tips
- Ensure consistent naming conventions
- Use standard currency formats
- Avoid special characters in product names
- Keep category names consistent across entries
