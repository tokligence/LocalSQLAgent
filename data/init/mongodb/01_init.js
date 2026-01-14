// MongoDB 测试数据初始化

// 切换到benchmark数据库
db = db.getSiblingDB('benchmark');

// 创建用户集合
db.users.insertMany([
    { _id: 1, name: "Alice", email: "alice@example.com", age: 28, department_id: 1, salary: 75000, created_at: new Date("2024-01-01") },
    { _id: 2, name: "Bob", email: "bob@example.com", age: 35, department_id: 1, salary: 95000, created_at: new Date("2024-01-02") },
    { _id: 3, name: "Charlie", email: "charlie@example.com", age: 42, department_id: 2, salary: 85000, created_at: new Date("2024-01-03") },
    { _id: 4, name: "Diana", email: "diana@example.com", age: 31, department_id: 2, salary: 72000, created_at: new Date("2024-01-04") },
    { _id: 5, name: "Eve", email: "eve@example.com", age: 26, department_id: 3, salary: 65000, created_at: new Date("2024-01-05") },
    { _id: 6, name: "Frank", email: "frank@example.com", age: 45, department_id: 1, salary: 120000, created_at: new Date("2024-01-06") },
    { _id: 7, name: "Grace", email: "grace@example.com", age: 29, department_id: 4, salary: 55000, created_at: new Date("2024-01-07") },
    { _id: 8, name: "Henry", email: "henry@example.com", age: 38, department_id: 2, salary: 90000, created_at: new Date("2024-01-08") },
    { _id: 9, name: "Ivy", email: "ivy@example.com", age: 24, department_id: 3, salary: 58000, created_at: new Date("2024-01-09") },
    { _id: 10, name: "Jack", email: "jack@example.com", age: 33, department_id: 1, salary: 88000, created_at: new Date("2024-01-10") }
]);

// 创建部门集合
db.departments.insertMany([
    { _id: 1, name: "Engineering", budget: 500000 },
    { _id: 2, name: "Sales", budget: 300000 },
    { _id: 3, name: "Marketing", budget: 200000 },
    { _id: 4, name: "HR", budget: 100000 }
]);

// 创建产品集合
db.products.insertMany([
    { _id: 1, name: "Laptop Pro", category: "Electronics", price: 1299.99, stock: 50 },
    { _id: 2, name: "Wireless Mouse", category: "Electronics", price: 29.99, stock: 200 },
    { _id: 3, name: "Office Chair", category: "Furniture", price: 249.99, stock: 30 },
    { _id: 4, name: "Standing Desk", category: "Furniture", price: 599.99, stock: 15 },
    { _id: 5, name: "Monitor 27\"", category: "Electronics", price: 399.99, stock: 75 },
    { _id: 6, name: "Keyboard", category: "Electronics", price: 89.99, stock: 150 },
    { _id: 7, name: "Webcam HD", category: "Electronics", price: 79.99, stock: 100 },
    { _id: 8, name: "Desk Lamp", category: "Furniture", price: 45.99, stock: 80 }
]);

// 创建订单集合
db.orders.insertMany([
    { _id: 1, user_id: 1, amount: 1329.98, status: "completed", items: [{product_id: 1, quantity: 1}, {product_id: 2, quantity: 1}], created_at: new Date("2024-01-15T10:30:00") },
    { _id: 2, user_id: 2, amount: 599.99, status: "completed", items: [{product_id: 4, quantity: 1}], created_at: new Date("2024-01-16T14:20:00") },
    { _id: 3, user_id: 1, amount: 89.99, status: "pending", items: [{product_id: 6, quantity: 1}], created_at: new Date("2024-01-17T09:15:00") },
    { _id: 4, user_id: 3, amount: 249.99, status: "completed", items: [{product_id: 3, quantity: 1}], created_at: new Date("2024-01-18T16:45:00") },
    { _id: 5, user_id: 4, amount: 1699.98, status: "shipped", items: [{product_id: 1, quantity: 1}, {product_id: 5, quantity: 1}], created_at: new Date("2024-01-19T11:00:00") },
    { _id: 6, user_id: 5, amount: 79.99, status: "completed", items: [{product_id: 7, quantity: 1}], created_at: new Date("2024-01-20T13:30:00") },
    { _id: 7, user_id: 2, amount: 449.98, status: "pending", items: [{product_id: 5, quantity: 1}, {product_id: 8, quantity: 1}], created_at: new Date("2024-01-21T10:00:00") },
    { _id: 8, user_id: 6, amount: 2099.97, status: "completed", items: [{product_id: 1, quantity: 1}, {product_id: 5, quantity: 2}], created_at: new Date("2024-01-22T15:45:00") },
    { _id: 9, user_id: 3, amount: 29.99, status: "cancelled", items: [{product_id: 2, quantity: 1}], created_at: new Date("2024-01-23T09:30:00") },
    { _id: 10, user_id: 1, amount: 399.99, status: "completed", items: [{product_id: 5, quantity: 1}], created_at: new Date("2024-01-24T14:00:00") }
]);

// 创建索引
db.users.createIndex({ email: 1 }, { unique: true });
db.users.createIndex({ department_id: 1 });
db.orders.createIndex({ user_id: 1 });
db.orders.createIndex({ created_at: -1 });
db.products.createIndex({ category: 1 });

print("MongoDB benchmark database initialized successfully!");
