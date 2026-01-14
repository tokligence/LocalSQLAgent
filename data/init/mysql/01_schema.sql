-- MySQL 测试数据初始化

-- 用户表
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE,
    age INT,
    department_id INT,
    salary DECIMAL(10, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 部门表
CREATE TABLE departments (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    budget DECIMAL(12, 2)
);

-- 订单表
CREATE TABLE orders (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    amount DECIMAL(10, 2),
    status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- 产品表
CREATE TABLE products (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    category VARCHAR(100),
    price DECIMAL(10, 2),
    stock INT DEFAULT 0
);

-- 订单明细表
CREATE TABLE order_items (
    id INT AUTO_INCREMENT PRIMARY KEY,
    order_id INT,
    product_id INT,
    quantity INT,
    unit_price DECIMAL(10, 2),
    FOREIGN KEY (order_id) REFERENCES orders(id),
    FOREIGN KEY (product_id) REFERENCES products(id)
);

-- 插入测试数据 (与PostgreSQL相同)
INSERT INTO departments (name, budget) VALUES
('Engineering', 500000),
('Sales', 300000),
('Marketing', 200000),
('HR', 100000);

INSERT INTO users (name, email, age, department_id, salary) VALUES
('Alice', 'alice@example.com', 28, 1, 75000),
('Bob', 'bob@example.com', 35, 1, 95000),
('Charlie', 'charlie@example.com', 42, 2, 85000),
('Diana', 'diana@example.com', 31, 2, 72000),
('Eve', 'eve@example.com', 26, 3, 65000),
('Frank', 'frank@example.com', 45, 1, 120000),
('Grace', 'grace@example.com', 29, 4, 55000),
('Henry', 'henry@example.com', 38, 2, 90000),
('Ivy', 'ivy@example.com', 24, 3, 58000),
('Jack', 'jack@example.com', 33, 1, 88000);

INSERT INTO products (name, category, price, stock) VALUES
('Laptop Pro', 'Electronics', 1299.99, 50),
('Wireless Mouse', 'Electronics', 29.99, 200),
('Office Chair', 'Furniture', 249.99, 30),
('Standing Desk', 'Furniture', 599.99, 15),
('Monitor 27"', 'Electronics', 399.99, 75),
('Keyboard', 'Electronics', 89.99, 150),
('Webcam HD', 'Electronics', 79.99, 100),
('Desk Lamp', 'Furniture', 45.99, 80);

INSERT INTO orders (user_id, amount, status, created_at) VALUES
(1, 1329.98, 'completed', '2024-01-15 10:30:00'),
(2, 599.99, 'completed', '2024-01-16 14:20:00'),
(1, 89.99, 'pending', '2024-01-17 09:15:00'),
(3, 249.99, 'completed', '2024-01-18 16:45:00'),
(4, 1699.98, 'shipped', '2024-01-19 11:00:00'),
(5, 79.99, 'completed', '2024-01-20 13:30:00'),
(2, 449.98, 'pending', '2024-01-21 10:00:00'),
(6, 2099.97, 'completed', '2024-01-22 15:45:00'),
(3, 29.99, 'cancelled', '2024-01-23 09:30:00'),
(1, 399.99, 'completed', '2024-01-24 14:00:00');

INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES
(1, 1, 1, 1299.99),
(1, 2, 1, 29.99),
(2, 4, 1, 599.99),
(3, 6, 1, 89.99),
(4, 3, 1, 249.99),
(5, 1, 1, 1299.99),
(5, 5, 1, 399.99),
(6, 7, 1, 79.99),
(7, 5, 1, 399.99),
(7, 8, 1, 45.99),
(8, 1, 1, 1299.99),
(8, 5, 1, 399.99),
(8, 5, 1, 399.99),
(9, 2, 1, 29.99),
(10, 5, 1, 399.99);
