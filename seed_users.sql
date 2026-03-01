-- Seed ~15 placeholder users with friendships and characteristics.
-- Run after schema (e.g. mysql ... align < seed_users.sql)
-- User ids are of the form 900001google .. 900015google to avoid colliding with real accounts.

USE align;

-- 15 dummy users (id = provider_sub + provider, e.g. 900001google)
INSERT INTO users (id, provider, provider_sub, email, is_hidden, user_settings, privacy_settings) VALUES
('900001google', 'google', '900001', 'alex.rivera@example.com', FALSE, '{}', '{"showEmail":false,"showAge":false,"showBirthday":false,"showPersonality":true}'),
('900002google', 'google', '900002', 'blake.chen@example.com', FALSE, '{}', '{"showEmail":false,"showAge":false,"showBirthday":false,"showPersonality":true}'),
('900003google', 'google', '900003', 'casey.morgan@example.com', FALSE, '{}', '{"showEmail":false,"showAge":false,"showBirthday":false,"showPersonality":true}'),
('900004google', 'google', '900004', 'drew.taylor@example.com', FALSE, '{}', '{"showEmail":false,"showAge":false,"showBirthday":false,"showPersonality":true}'),
('900005google', 'google', '900005', 'emery.lee@example.com', FALSE, '{}', '{"showEmail":false,"showAge":false,"showBirthday":false,"showPersonality":true}'),
('900006google', 'google', '900006', 'finley.wright@example.com', FALSE, '{}', '{"showEmail":false,"showAge":false,"showBirthday":false,"showPersonality":true}'),
('900007google', 'google', '900007', 'gray.hall@example.com', FALSE, '{}', '{"showEmail":false,"showAge":false,"showBirthday":false,"showPersonality":true}'),
('900008google', 'google', '900008', 'jordan.kim@example.com', FALSE, '{}', '{"showEmail":false,"showAge":false,"showBirthday":false,"showPersonality":true}'),
('900009google', 'google', '900009', 'quinn.patel@example.com', FALSE, '{}', '{"showEmail":false,"showAge":false,"showBirthday":false,"showPersonality":true}'),
('900010google', 'google', '900010', 'reese.brooks@example.com', FALSE, '{}', '{"showEmail":false,"showAge":false,"showBirthday":false,"showPersonality":true}'),
('900011google', 'google', '900011', 'sage.clark@example.com', FALSE, '{}', '{"showEmail":false,"showAge":false,"showBirthday":false,"showPersonality":true}'),
('900012google', 'google', '900012', 'taylor.evans@example.com', FALSE, '{}', '{"showEmail":false,"showAge":false,"showBirthday":false,"showPersonality":true}'),
('900013google', 'google', '900013', 'rowan.nguyen@example.com', FALSE, '{}', '{"showEmail":false,"showAge":false,"showBirthday":false,"showPersonality":true}'),
('900014google', 'google', '900014', 'skyler.king@example.com', FALSE, '{}', '{"showEmail":false,"showAge":false,"showBirthday":false,"showPersonality":true}'),
('900015google', 'google', '900015', 'morgan.bell@example.com', FALSE, '{}', '{"showEmail":false,"showAge":false,"showBirthday":false,"showPersonality":true}')
ON DUPLICATE KEY UPDATE email = VALUES(email), updated_at = CURRENT_TIMESTAMP;

-- Friends: bidirectional pairs. (1-2, 1-3, 2-3, 2-4, 3-4, 3-5, 4-5, 4-6, 5-6, 6-7, 7-8, 8-9, 9-10, 10-11, 11-12, 12-13, 1-5, 2-6, 3-7)
INSERT IGNORE INTO friends (user_id, friend_id) VALUES
('900001google', '900002google'), ('900002google', '900001google'),
('900001google', '900003google'), ('900003google', '900001google'),
('900002google', '900003google'), ('900003google', '900002google'),
('900002google', '900004google'), ('900004google', '900002google'),
('900003google', '900004google'), ('900004google', '900003google'),
('900003google', '900005google'), ('900005google', '900003google'),
('900004google', '900005google'), ('900005google', '900004google'),
('900004google', '900006google'), ('900006google', '900004google'),
('900005google', '900006google'), ('900006google', '900005google'),
('900006google', '900007google'), ('900007google', '900006google'),
('900007google', '900008google'), ('900008google', '900007google'),
('900008google', '900009google'), ('900009google', '900008google'),
('900009google', '900010google'), ('900010google', '900009google'),
('900010google', '900011google'), ('900011google', '900010google'),
('900011google', '900012google'), ('900012google', '900011google'),
('900012google', '900013google'), ('900013google', '900012google'),
('900001google', '900005google'), ('900005google', '900001google'),
('900002google', '900006google'), ('900006google', '900002google'),
('900003google', '900007google'), ('900007google', '900003google');

-- Characteristics: a few text traits per user (value_text only; personality_vector omitted for seed simplicity)
INSERT INTO characteristics (user_id, trait_key, value_text, is_public, manually_overridden) VALUES
('900001google', 'star_sign', 'Aries', TRUE, TRUE),
('900001google', 'myers_briggs', 'ENFP', TRUE, TRUE),
('900001google', 'moral_foundation', 'Care and fairness', TRUE, TRUE),
('900002google', 'star_sign', 'Taurus', TRUE, TRUE),
('900002google', 'myers_briggs', 'ISTJ', TRUE, TRUE),
('900002google', 'political_leaning', 'Moderate', TRUE, TRUE),
('900003google', 'star_sign', 'Gemini', TRUE, TRUE),
('900003google', 'attachment_style', 'Secure', TRUE, TRUE),
('900003google', 'love_language', 'Words of affirmation', TRUE, TRUE),
('900004google', 'star_sign', 'Cancer', TRUE, TRUE),
('900004google', 'enneagram_type', 'Type 2', TRUE, TRUE),
('900004google', 'humor_style', 'Self-deprecating', TRUE, TRUE),
('900005google', 'star_sign', 'Leo', TRUE, TRUE),
('900005google', 'myers_briggs', 'ENTP', TRUE, TRUE),
('900005google', 'conflict_style', 'Collaborative', TRUE, TRUE),
('900006google', 'star_sign', 'Virgo', TRUE, TRUE),
('900006google', 'learning_style', 'Visual', TRUE, TRUE),
('900006google', 'core_values', 'Integrity, growth', TRUE, TRUE),
('900007google', 'star_sign', 'Libra', TRUE, TRUE),
('900007google', 'myers_briggs', 'INFJ', TRUE, TRUE),
('900007google', 'spirituality', 'Secular humanist', TRUE, TRUE),
('900008google', 'star_sign', 'Scorpio', TRUE, TRUE),
('900008google', 'political_leaning', 'Progressive', TRUE, TRUE),
('900008google', 'communication_style', 'Direct', TRUE, TRUE),
('900009google', 'star_sign', 'Sagittarius', TRUE, TRUE),
('900009google', 'chronotype', 'Night owl', TRUE, TRUE),
('900009google', 'risk_tolerance', 'Moderate', TRUE, TRUE),
('900010google', 'star_sign', 'Capricorn', TRUE, TRUE),
('900010google', 'myers_briggs', 'INTJ', TRUE, TRUE),
('900010google', 'life_philosophy', 'Stoic leanings', TRUE, TRUE),
('900011google', 'star_sign', 'Aquarius', TRUE, TRUE),
('900011google', 'attachment_style', 'Anxious', TRUE, TRUE),
('900011google', 'creativity_style', 'Ideation', TRUE, TRUE),
('900012google', 'star_sign', 'Pisces', TRUE, TRUE),
('900012google', 'myers_briggs', 'ISFP', TRUE, TRUE),
('900012google', 'empathy_style', 'High', TRUE, TRUE),
('900013google', 'star_sign', 'Aries', TRUE, TRUE),
('900013google', 'moral_foundation', 'Liberty and authority', TRUE, TRUE),
('900013google', 'leadership_style', 'Servant', TRUE, TRUE),
('900014google', 'star_sign', 'Taurus', TRUE, TRUE),
('900014google', 'myers_briggs', 'ESTP', TRUE, TRUE),
('900014google', 'humor_style', 'Dry', TRUE, TRUE),
('900015google', 'star_sign', 'Gemini', TRUE, TRUE),
('900015google', 'love_language', 'Quality time', TRUE, TRUE),
('900015google', 'decision_style', 'Analytical', TRUE, TRUE)
ON DUPLICATE KEY UPDATE value_text = VALUES(value_text), is_public = VALUES(is_public), manually_overridden = VALUES(manually_overridden);

-- ---------------------------------------------------------------------------
-- Example: adding one person's everything (user + friends + characteristics)
-- ---------------------------------------------------------------------------
-- 1) One user
INSERT INTO users (id, provider, provider_sub, email, is_hidden, user_settings, privacy_settings) VALUES
('900099google', 'google', '900099', 'single.user@example.com', FALSE, '{}', '{"showEmail":false,"showAge":false,"showBirthday":false,"showPersonality":true}')
ON DUPLICATE KEY UPDATE email = VALUES(email), updated_at = CURRENT_TIMESTAMP;

-- 2) Friends (bidirectional: make them friends with 900001 and 900002)
INSERT IGNORE INTO friends (user_id, friend_id) VALUES
('900099google', '900001google'), ('900001google', '900099google'),
('900099google', '900002google'), ('900002google', '900099google');

-- 3) Characteristics for that person
INSERT INTO characteristics (user_id, trait_key, value_text, is_public, manually_overridden) VALUES
('900099google', 'star_sign', 'Libra', TRUE, TRUE),
('900099google', 'myers_briggs', 'INFP', TRUE, TRUE),
('900099google', 'love_language', 'Acts of service', TRUE, TRUE),
('900099google', 'moral_foundation', 'Care and fairness', TRUE, TRUE)
ON DUPLICATE KEY UPDATE value_text = VALUES(value_text), is_public = VALUES(is_public), manually_overridden = VALUES(manually_overridden);
