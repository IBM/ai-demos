import psycopg2
import csv
import os

# Database connection parameters
DB_HOST = os.getenv('DB_HOST', 'techmart-postgresql')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'techmart')
DB_USER = os.getenv('DB_USER', 'techmart')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'techmart123')

def load_orders_from_csv():
    """Load orders from CSV file"""
    orders = []
    csv_path = '/data/orders.csv'

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            orders.append(row)

    return orders

def insert_orders(orders):
    """Insert orders into PostgreSQL database"""
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )

    cursor = conn.cursor()

    # Insert orders
    insert_query = """
        INSERT INTO orders (
            order_id, customer_email, product_name, category,
            order_date, delivery_date, price, status, is_opened
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (order_id) DO UPDATE SET
            customer_email = EXCLUDED.customer_email,
            product_name = EXCLUDED.product_name,
            category = EXCLUDED.category,
            order_date = EXCLUDED.order_date,
            delivery_date = EXCLUDED.delivery_date,
            price = EXCLUDED.price,
            status = EXCLUDED.status,
            is_opened = EXCLUDED.is_opened
    """

    for order in orders:
        cursor.execute(insert_query, (
            order['order_id'],
            order['customer_email'],
            order['product_name'],
            order['category'],
            order['order_date'],
            order['delivery_date'],
            float(order['price']),
            order['status'],
            order['is_opened']
        ))

    conn.commit()
    print(f"✅ Inserted {len(orders)} orders")

    cursor.close()
    conn.close()

if __name__ == "__main__":
    print("📊 Loading orders from CSV...")
    orders = load_orders_from_csv()
    print(f"   Found {len(orders)} orders")

    print("💾 Inserting orders into database...")
    insert_orders(orders)

    print("✅ Database initialization complete!")
