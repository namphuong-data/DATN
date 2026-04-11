
CREATE OR REPLACE VIEW bronze_taxi AS

SELECT * FROM delta_scan('s3://lakehouse/bronze/all/');

