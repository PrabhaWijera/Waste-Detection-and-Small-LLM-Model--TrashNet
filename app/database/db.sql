CREATE DATABASE waste_management;

USE waste_management;

CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(100) UNIQUE,
    email VARCHAR(255),
    password VARCHAR(255),
    role ENUM('normal', 'urban') DEFAULT 'normal'
);

CREATE TABLE waste_records (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    image_path VARCHAR(255),  -- store file path, not image
    waste_type VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE impact_reports (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    month YEAR,
    waste_saved FLOAT,
    co2_reduced FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE models (
    id INT AUTO_INCREMENT PRIMARY KEY,
    model_name VARCHAR(100),
    version VARCHAR(50),
    file_path VARCHAR(255),   -- store model file path
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
