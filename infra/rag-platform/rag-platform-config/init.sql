-- Rag Platform — owned MySQL bootstrap
--
-- Bind-mounted over /data/application/init.sql and executed by the mysql
-- server's --init-file on every start (see the base compose file's
-- `--init-file /data/application/init.sql` command flag). Must therefore
-- stay idempotent and cheap: only IF NOT EXISTS DDL, no data statements.
--
-- Why this file has to exist: the backend never creates its own schema.
-- api.db.db_models.init_database_tables creates TABLES inside an already
-- existing database, and no CREATE DATABASE statement appears anywhere in
-- the backend's Python or Go sources. Renaming the schema via MYSQL_DBNAME
-- alone would leave the application pointed at a database that nothing
-- creates.
--
-- Character set and collation match the server flags the base compose
-- file passes (--character-set-server=utf8mb4
-- --collation-server=utf8mb4_unicode_ci), so the schema does not silently
-- diverge from the connection defaults.
--
-- Keep in step with MYSQL_DBNAME in .env.rag-platform.

CREATE DATABASE IF NOT EXISTS rag_platform
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE rag_platform;
