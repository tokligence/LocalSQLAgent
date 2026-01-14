-- 电商场景测试数据
-- 包含：用户、商品、订单、评价等真实业务场景

-- 创建表结构
CREATE TABLE IF NOT EXISTS customers (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    registration_date DATE,
    vip_level INTEGER DEFAULT 0,  -- 0: 普通, 1: 银牌, 2: 金牌, 3: 钻石
    total_spent DECIMAL(12,2) DEFAULT 0,
    city VARCHAR(50),
    country VARCHAR(50)
);

CREATE TABLE IF NOT EXISTS product_categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) NOT NULL,
    parent_category_id INTEGER REFERENCES product_categories(id)
);

CREATE TABLE IF NOT EXISTS products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    category_id INTEGER REFERENCES product_categories(id),
    price DECIMAL(10,2) NOT NULL,
    cost DECIMAL(10,2),  -- 成本
    stock_quantity INTEGER DEFAULT 0,
    rating DECIMAL(3,2),  -- 平均评分
    launch_date DATE,
    is_active BOOLEAN DEFAULT true
);

CREATE TABLE IF NOT EXISTS orders (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(id),
    order_date TIMESTAMP NOT NULL,
    status VARCHAR(20),  -- pending, paid, shipped, delivered, cancelled
    total_amount DECIMAL(10,2),
    discount_amount DECIMAL(10,2) DEFAULT 0,
    shipping_fee DECIMAL(10,2) DEFAULT 0,
    payment_method VARCHAR(20)  -- credit_card, paypal, alipay, wechat
);

CREATE TABLE IF NOT EXISTS order_items (
    id SERIAL PRIMARY KEY,
    order_id INTEGER REFERENCES orders(id),
    product_id INTEGER REFERENCES products(id),
    quantity INTEGER NOT NULL,
    unit_price DECIMAL(10,2) NOT NULL,
    discount_percent DECIMAL(5,2) DEFAULT 0
);

CREATE TABLE IF NOT EXISTS product_reviews (
    id SERIAL PRIMARY KEY,
    product_id INTEGER REFERENCES products(id),
    customer_id INTEGER REFERENCES customers(id),
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    review_text TEXT,
    review_date TIMESTAMP,
    is_verified_purchase BOOLEAN DEFAULT false
);

CREATE TABLE IF NOT EXISTS shopping_cart (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(id),
    product_id INTEGER REFERENCES products(id),
    quantity INTEGER DEFAULT 1,
    added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 插入测试数据

-- 产品类别
INSERT INTO product_categories (name, parent_category_id) VALUES
('电子产品', NULL),
('手机', 1),
('电脑', 1),
('服装', NULL),
('男装', 4),
('女装', 4),
('图书', NULL),
('家居', NULL);

-- 客户数据（不同消费水平）
INSERT INTO customers (name, email, registration_date, vip_level, total_spent, city, country) VALUES
('张三', 'zhang3@email.com', '2023-01-15', 3, 125000.00, '上海', '中国'),
('李四', 'li4@email.com', '2023-03-20', 2, 45000.00, '北京', '中国'),
('王五', 'wang5@email.com', '2023-06-10', 1, 8500.00, '广州', '中国'),
('赵六', 'zhao6@email.com', '2024-01-05', 0, 2500.00, '深圳', '中国'),
('Emma Johnson', 'emma@email.com', '2023-02-28', 2, 35000.00, 'New York', 'USA'),
('John Smith', 'john@email.com', '2023-07-15', 1, 12000.00, 'London', 'UK'),
('Marie Dubois', 'marie@email.com', '2023-09-01', 0, 3200.00, 'Paris', 'France'),
('Hans Mueller', 'hans@email.com', '2023-11-20', 3, 98000.00, 'Berlin', 'Germany'),
('Yuki Tanaka', 'yuki@email.com', '2024-02-10', 1, 15000.00, 'Tokyo', 'Japan'),
('陈七', 'chen7@email.com', '2024-03-01', 0, 800.00, '成都', '中国');

-- 商品数据（不同类别和价格区间）
INSERT INTO products (name, category_id, price, cost, stock_quantity, rating, launch_date, is_active) VALUES
('iPhone 15 Pro', 2, 8999.00, 6500.00, 50, 4.8, '2023-09-15', true),
('Samsung Galaxy S24', 2, 6999.00, 5000.00, 80, 4.6, '2024-01-15', true),
('MacBook Pro 16"', 3, 19999.00, 15000.00, 20, 4.9, '2023-11-01', true),
('ThinkPad X1 Carbon', 3, 12999.00, 9000.00, 35, 4.5, '2023-06-01', true),
('男士休闲衬衫', 5, 299.00, 120.00, 200, 4.2, '2024-03-01', true),
('女士连衣裙', 6, 599.00, 250.00, 150, 4.4, '2024-03-15', true),
('Python编程入门', 7, 59.00, 30.00, 500, 4.7, '2023-01-01', true),
('智能台灯', 8, 199.00, 80.00, 300, 4.1, '2023-08-01', true),
('无线耳机 AirPods Pro', 1, 1999.00, 1200.00, 100, 4.6, '2023-05-01', true),
('小米手环8', 1, 299.00, 150.00, 1000, 4.3, '2024-01-01', true);

-- 订单数据（不同时间段和状态）
INSERT INTO orders (customer_id, order_date, status, total_amount, discount_amount, shipping_fee, payment_method) VALUES
(1, '2024-01-10 10:30:00', 'delivered', 8999.00, 200.00, 0, 'credit_card'),
(1, '2024-01-15 14:20:00', 'delivered', 19999.00, 500.00, 0, 'credit_card'),
(2, '2024-01-20 09:15:00', 'delivered', 6999.00, 100.00, 0, 'alipay'),
(3, '2024-02-01 16:45:00', 'delivered', 299.00, 0, 10.00, 'wechat'),
(4, '2024-02-10 11:30:00', 'shipped', 59.00, 0, 15.00, 'paypal'),
(5, '2024-02-15 13:00:00', 'delivered', 12999.00, 300.00, 0, 'credit_card'),
(6, '2024-02-20 10:00:00', 'pending', 599.00, 50.00, 20.00, 'paypal'),
(7, '2024-03-01 15:30:00', 'paid', 199.00, 0, 10.00, 'credit_card'),
(8, '2024-03-05 09:00:00', 'delivered', 1999.00, 100.00, 0, 'credit_card'),
(1, '2024-03-10 14:00:00', 'delivered', 299.00, 0, 0, 'alipay'),
(2, '2024-03-15 16:00:00', 'cancelled', 299.00, 0, 10.00, 'wechat'),
(9, '2024-03-20 11:00:00', 'processing', 8999.00, 0, 0, 'credit_card');

-- 订单明细
INSERT INTO order_items (order_id, product_id, quantity, unit_price, discount_percent) VALUES
(1, 1, 1, 8999.00, 2.0),
(2, 3, 1, 19999.00, 2.5),
(3, 2, 1, 6999.00, 1.5),
(4, 5, 1, 299.00, 0),
(5, 7, 1, 59.00, 0),
(6, 4, 1, 12999.00, 2.3),
(7, 6, 1, 599.00, 8.0),
(8, 8, 1, 199.00, 0),
(9, 9, 1, 1999.00, 5.0),
(10, 10, 1, 299.00, 0),
(11, 5, 1, 299.00, 0),
(12, 1, 1, 8999.00, 0);

-- 商品评价
INSERT INTO product_reviews (product_id, customer_id, rating, review_text, review_date, is_verified_purchase) VALUES
(1, 1, 5, '非常好用，性能强大！', '2024-01-20 10:00:00', true),
(1, 2, 4, '价格有点贵，但质量不错', '2024-02-01 14:00:00', false),
(2, 2, 5, '安卓机皇，推荐！', '2024-01-25 09:00:00', true),
(3, 1, 5, '专业工作必备', '2024-01-20 15:00:00', true),
(5, 3, 4, '质量不错，很舒适', '2024-02-05 11:00:00', true),
(7, 4, 5, '入门好书', '2024-02-15 16:00:00', true),
(9, 8, 4, '音质很好', '2024-03-10 10:00:00', true);

-- 购物车数据
INSERT INTO shopping_cart (customer_id, product_id, quantity, added_date) VALUES
(1, 2, 1, '2024-03-25 10:00:00'),
(1, 9, 2, '2024-03-25 10:05:00'),
(3, 5, 3, '2024-03-24 15:00:00'),
(4, 7, 1, '2024-03-23 09:00:00');