-- 1. Courses (no dependencies)
CREATE TABLE "Courses" (
  id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  "createdAt" TIMESTAMPTZ NOT NULL DEFAULT now(),
  "courseName" TEXT DEFAULT ''
);

-- 2. coursematerials (no dependencies)
CREATE TABLE coursematerials (
  id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  "fileName" TEXT NOT NULL DEFAULT '',
  "parsed" BOOLEAN NOT NULL DEFAULT FALSE
);

-- 3. Concepts (depends on Courses)
CREATE TABLE "Concepts" (
  id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  "conceptName" TEXT DEFAULT '',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  courseid BIGINT REFERENCES "Courses"(id),
  "materialIds" JSONB
);

-- 4. conceptlinks (depends on Concepts and Courses)
CREATE TABLE conceptlinks (
  id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  sourceconceptid BIGINT NOT NULL REFERENCES "Concepts"(id),
  targetconceptid BIGINT NOT NULL REFERENCES "Concepts"(id),
  linktype TEXT DEFAULT 'related',
  courseid BIGINT REFERENCES "Courses"(id)
);