# coding=utf-8
import pandas as pd
from pprint import pprint
import xlrd
from mlxtend.frequent_patterns import apriori, association_rules


def read_file(file):
    """
    :param file: purchase checks source file
    :return: sales data frame, list with unique products
    """
    sales = pd.read_excel(file, sheet_name='Чеки')
    print (sales.columns)
    sales = sales.astype({"Название товаров": str})
    sales["Название товаров"] = sales["Название товаров"]\
        .str.strip()\
        .str.replace("   ", " ")\
        .str.replace("  ", " ")
    unique_products = sales.loc[1:, 'Название товаров'].unique()

    return sales, unique_products


def create_trans_dict(sales_table):
    """
    :param sales_table: sales data frame
    :return: prepared table with purchases (0/1) for associative analysis
    """
    transactions = (sales_table
                    .groupby(['#', 'Название товаров'])['Кол-во товаров']
                    .sum().unstack().reset_index().fillna(0)
                    .set_index('#'))

    transactions = transactions.applymap(hot_encode)
    transactions = transactions.fillna(0)
    return transactions


def hot_encode(x):
    """
    :param x: number of products purchased
    :return: boolean value (0/1) for the product
    """
    if x <= 0:
        return False
    if x >= 1:
        return True


def format_output(rules_table):
    """
    :param rules_table: associative rules table
    :return: export formatted table with rules
    """
    rules_table = rules_table.astype({"antecedents": str, "consequents": str})
    type_set = ["antecedents", "consequents"]

    for type in type_set:
        rules_table[type] = rules_table[type] \
            .str.replace("frozenset\({", "") \
            .str.replace("}\)", "") \
            .str.replace(" \+", " + ") \
            .str.replace(".", "")\
            .str.replace("'", "")\
            .str.lower() \
            .str.split(',')

    rules_table = rules_table.sort_values(['confidence', 'lift'], ascending=[False, False])
    export_excel = rules_table.to_excel(r'C:\Users\Admin\Desktop\Asociation_rules.xlsx', header=True)

    return export_excel


if __name__== '__main__':
    FILE = r'products_sale.xls'
    sales, products = read_file(FILE)
    df_sales = create_trans_dict(sales)
    df_sales = df_sales[df_sales.columns.drop(['nan'])]

    frq_items = apriori(df_sales, min_support=0.002, use_colnames=True)
    rules = association_rules(frq_items, metric="lift", min_threshold=1)

    format_output(rules)