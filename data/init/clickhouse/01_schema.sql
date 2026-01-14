-- ClickHouse 测试数据初始化 (分析型场景)

-- 用户表
CREATE TABLE IF NOT EXISTS users (
    id UInt32,
    name String,
    email String,
    age UInt8,
    department_id UInt32,
    salary Decimal(10, 2),
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
ORDER BY id;

-- 部门表
CREATE TABLE IF NOT EXISTS departments (
    id UInt32,
    name String,
    budget Decimal(12, 2)
) ENGINE = MergeTree()
ORDER BY id;

-- 订单表 (大数据分析场景)
CREATE TABLE IF NOT EXISTS orders (
    id UInt64,
    user_id UInt32,
    amount Decimal(10, 2),
    status String,
    created_at DateTime
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(created_at)
ORDER BY (created_at, id);

-- 产品表
CREATE TABLE IF NOT EXISTS products (
    id UInt32,
    name String,
    category String,
    price Decimal(10, 2),
    stock UInt32
) ENGINE = MergeTree()
ORDER BY id;

-- 订单明细表
CREATE TABLE IF NOT EXISTS order_items (
    id UInt64,
    order_id UInt64,
    product_id UInt32,
    quantity UInt32,
    unit_price Decimal(10, 2)
) ENGINE = MergeTree()
ORDER BY (order_id, id);

-- 事件日志表 (ClickHouse典型场景)
CREATE TABLE IF NOT EXISTS events (
    event_id UUID DEFAULT generateUUIDv4(),
    user_id UInt32,
    event_type String,
    event_data String,
    timestamp DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY toYYYYMMDD(timestamp)
ORDER BY (timestamp, event_id);

-- 插入测试数据
INSERT INTO departments (id, name, budget) VALUES
(1, 'Engineering', 500000),
(2, 'Sales', 300000),
(3, 'Marketing', 200000),
(4, 'HR', 100000);

INSERT INTO users (id, name, email, age, department_id, salary) VALUES
(1, 'Alice', 'alice@example.com', 28, 1, 75000),
(2, 'Bob', 'bob@example.com', 35, 1, 95000),
(3, 'Charlie', 'charlie@example.com', 42, 2, 85000),
(4, 'Diana', 'diana@example.com', 31, 2, 72000),
(5, 'Eve', 'eve@example.com', 26, 3, 65000),
(6, 'Frank', 'frank@example.com', 45, 1, 120000),
(7, 'Grace', 'grace@example.com', 29, 4, 55000),
(8, 'Henry', 'henry@example.com', 38, 2, 90000),
(9, 'Ivy', 'ivy@example.com', 24, 3, 58000),
(10, 'Jack', 'jack@example.com', 33, 1, 88000);

INSERT INTO products (id, name, category, price, stock) VALUES
(1, 'Laptop Pro', 'Electronics', 1299.99, 50),
(2, 'Wireless Mouse', 'Electronics', 29.99, 200),
(3, 'Office Chair', 'Furniture', 249.99, 30),
(4, 'Standing Desk', 'Furniture', 599.99, 15),
(5, 'Monitor 27"', 'Electronics', 399.99, 75),
(6, 'Keyboard', 'Electronics', 89.99, 150),
(7, 'Webcam HD', 'Electronics', 79.99, 100),
(8, 'Desk Lamp', 'Furniture', 45.99, 80);

INSERT INTO orders (id, user_id, amount, status, created_at) VALUES
(1, 1, 1329.98, 'completed', '2024-01-15 10:30:00'),
(2, 2, 599.99, 'completed', '2024-01-16 14:20:00'),
(3, 1, 89.99, 'pending', '2024-01-17 09:15:00'),
(4, 3, 249.99, 'completed', '2024-01-18 16:45:00'),
(5, 4, 1699.98, 'shipped', '2024-01-19 11:00:00'),
(6, 5, 79.99, 'completed', '2024-01-20 13:30:00'),
(7, 2, 449.98, 'pending', '2024-01-21 10:00:00'),
(8, 6, 2099.97, 'completed', '2024-01-22 15:45:00'),
(9, 3, 29.99, 'cancelled', '2024-01-23 09:30:00'),
(10, 1, 399.99, 'completed', '2024-01-24 14:00:00');

INSERT INTO order_items (id, order_id, product_id, quantity, unit_price) VALUES
(1, 1, 1, 1, 1299.99),
(2, 1, 2, 1, 29.99),
(3, 2, 4, 1, 599.99),
(4, 3, 6, 1, 89.99),
(5, 4, 3, 1, 249.99),
(6, 5, 1, 1, 1299.99),
(7, 5, 5, 1, 399.99),
(8, 6, 7, 1, 79.99),
(9, 7, 5, 1, 399.99),
(10, 7, 8, 1, 45.99),
(11, 8, 1, 1, 1299.99),
(12, 8, 5, 1, 399.99),
(13, 8, 5, 1, 399.99),
(14, 9, 2, 1, 29.99),
(15, 10, 5, 1, 399.99);

-- 插入一些事件日志数据
INSERT INTO events (user_id, event_type, event_data, timestamp) VALUES
(1, 'login', '{"ip": "192.168.1.1"}', '2024-01-15 10:00:00'),
(1, 'purchase', '{"order_id": 1}', '2024-01-15 10:30:00'),
(2, 'login', '{"ip": "192.168.1.2"}', '2024-01-16 14:00:00'),
(2, 'purchase', '{"order_id": 2}', '2024-01-16 14:20:00'),
(3, 'login', '{"ip": "192.168.1.3"}', '2024-01-18 16:30:00'),
(3, 'purchase', '{"order_id": 4}', '2024-01-18 16:45:00');
