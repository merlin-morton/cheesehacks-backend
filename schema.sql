-- Cheesehacks 2026 Database Schema
-- Run this to create the database and tables

CREATE DATABASE IF NOT EXISTS cheesehacks;
USE cheesehacks;

-- Users table: core identity + profile + personality + privacy
CREATE TABLE IF NOT EXISTS users (
    id VARCHAR(255) PRIMARY KEY COMMENT 'providerSub + provider concatenated',
    provider VARCHAR(32) NOT NULL,
    provider_sub VARCHAR(255) NOT NULL,
    email VARCHAR(255) NOT NULL,
    -- Personality & profile
    personality_vector BLOB NULL COMMENT 'Serialized personality/quiz result vector',
    birthday DATE NULL,
    age INT NULL,
    -- User preferences (JSON for flexibility: theme, notifications, etc.)
    user_settings JSON NULL DEFAULT ('{}'),
    -- Privacy: whether account appears in lookups
    is_hidden BOOLEAN NOT NULL DEFAULT FALSE,
    -- What to show when profile is looked up (JSON: showEmail, showAge, showBirthday, showPersonality)
    privacy_settings JSON NULL DEFAULT ('{"showEmail":false,"showAge":false,"showBirthday":false,"showPersonality":true}'),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uk_provider_sub (provider, provider_sub)
);

-- Friends: many-to-many relationship
CREATE TABLE IF NOT EXISTS friends (
    user_id VARCHAR(255) NOT NULL,
    friend_id VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, friend_id),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (friend_id) REFERENCES users(id) ON DELETE CASCADE,
    CHECK (user_id != friend_id)
);

-- Quiz responses: store per-question responses before final submit (for getQuestion context)
CREATE TABLE IF NOT EXISTS quiz_responses (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    question_id VARCHAR(64) NOT NULL,
    response_data JSON NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    UNIQUE KEY uk_user_question (user_id, question_id)
);

-- Diagnostics: user-specific diagnostic data
CREATE TABLE IF NOT EXISTS diagnostics (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    diagnostic_data JSON NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
