import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
import re

# Scrape 4 categories
category_urls = {
    'Mystery': 'https://books.toscrape.com/catalogue/category/books/mystery_3/index.html',
    'Historical Fiction': 'https://books.toscrape.com/catalogue/category/books/historical-fiction_4/index.html',
    'Sequential Art': 'https://books.toscrape.com/catalogue/category/books/sequential-art_5/index.html',
    'Classics': 'https://books.toscrape.com/catalogue/category/books/classics_6/index.html'
}

books_data = []
word_to_num = {'One': 1, 'Two': 2, 'Three': 3, 'Four': 4, 'Five': 5}

for category, url in category_urls.items():
    while url:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        for article in soup.find_all('article', class_='product_pod'):
            title = article.h3.a['title']
            price_text = article.find('p', class_='price_color').text
            rating_text = article.p['class'][1] 
            stock_text = article.find('p', class_='instock availability').text.strip()
            
            price_gbp = float(re.sub(r'[^\d.]', '', price_text))
            rating = word_to_num.get(rating_text, 0)
            in_stock = 1 if 'in stock' in stock_text.lower() else 0
            
            books_data.append({
                'title': title,
                'price_gbp': price_gbp,
                'rating': rating,
                'in_stock': in_stock,
                'category_name': category
            })
        
        next_btn = soup.find('li', class_='next')
        if next_btn:
            next_url = next_btn.a['href']
            url = url.rsplit('/', 1)[0] + '/' + next_url
        else:
            url = None

df = pd.DataFrame(books_data)
CONVERSION_RATE = 105.50
df['price_inr'] = df['price_gbp'] * CONVERSION_RATE

conn = sqlite3.connect('zepto_books.db')
cursor = conn.cursor()

cursor.executescript('''
    DROP TABLE IF EXISTS books;
    DROP TABLE IF EXISTS categories;

    CREATE TABLE categories (
        category_id INTEGER PRIMARY KEY AUTOINCREMENT,
        category_name TEXT UNIQUE
    );

    CREATE TABLE books (
        book_id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        price_gbp REAL,
        price_inr REAL,
        rating INTEGER,
        in_stock INTEGER,
        category_id INTEGER,
        FOREIGN KEY (category_id) REFERENCES categories (category_id)
    );
''')

categories = df[['category_name']].drop_duplicates()
categories.to_sql('categories', conn, if_exists='append', index=False)

cat_df = pd.read_sql("SELECT * FROM categories", conn)
df = df.merge(cat_df, on='category_name', how='left')

books_to_insert = df[['title', 'price_gbp', 'price_inr', 'rating', 'in_stock', 'category_id']]
books_to_insert.to_sql('books', conn, if_exists='append', index=False)

queries = {
    "1. DISTINCT": "SELECT DISTINCT category_name FROM categories;",
    "2. IN, LIMIT": "SELECT title, rating FROM books WHERE rating IN (4, 5) LIMIT 5;",
    "3. BETWEEN, ORDER BY": "SELECT title, price_inr FROM books WHERE price_inr BETWEEN 2000 AND 5000 ORDER BY price_inr DESC LIMIT 5;",
    "4. SELECT / WHERE": "SELECT title, price_gbp FROM books WHERE in_stock = 1 AND price_gbp < 20 LIMIT 5;",
    "5. JOIN": '''
        SELECT c.category_name, b.title, b.rating 
        FROM categories c 
        JOIN books b ON c.category_id = b.category_id 
        WHERE b.rating = 5 
        LIMIT 5;
    '''
}

print("--- SQL QUERY OUTPUTS ---")
for name, query in queries.items():
    print(f"\n{name}:\n", pd.read_sql(query, conn))

sql_join_result = pd.read_sql(queries["5. JOIN"], conn)
books_memory = pd.read_sql("SELECT * FROM books", conn)
categories_memory = pd.read_sql("SELECT * FROM categories", conn)

pandas_merged = pd.merge(categories_memory, books_memory, on='category_id')
pandas_filtered = pandas_merged[pandas_merged['rating'] == 5][['category_name', 'title', 'rating']].head(5)

print("\n--- EQUIVALENCY CHECK ---")
print("SQL Output:\n", sql_join_result)
print("\nPandas Output:\n", pandas_filtered.reset_index(drop=True))

conn.close()