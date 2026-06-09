CREATE TABLE students (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(100),
    register_no VARCHAR(50) UNIQUE
);

CREATE TABLE attendance (
    id INT PRIMARY KEY AUTO_INCREMENT,
    student_id INT,
    date DATE,
    status VARCHAR(20),
    FOREIGN KEY(student_id) REFERENCES students(id)
);
DESCRIBE students;
USE sutdent;
SHOW TABLES;
DROP DATABASE student;
