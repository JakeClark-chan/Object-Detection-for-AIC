-- SELECT SUBSTR(
--     SUBSTR(image_name, INSTR(image_name, 'L'), LENGTH(image_name)),
--     INSTR(SUBSTR(image_name, INSTR(image_name, 'L'), LENGTH(image_name)), "/") + 1,
--     LENGTH(SUBSTR(image_name, INSTR(image_name, 'L'), LENGTH(image_name)))
-- ) AS new_image_name
-- FROM images
-- WHERE image_name LIKE '%/L__%_V___/%.jpg';

-- UPDATE images
-- SET image_name = SUBSTR(
--     SUBSTR(image_name, INSTR(image_name, 'L'), LENGTH(image_name)),
--     INSTR(SUBSTR(image_name, INSTR(image_name, 'L'), LENGTH(image_name)), "/") + 1,
--     LENGTH(SUBSTR(image_name, INSTR(image_name, 'L'), LENGTH(image_name)))
-- )
-- WHERE image_name LIKE '%/L__%_V___/%.jpg';

-- SELECT image_name FROM images WHERE object_color IS NOT NULL; 

-- Update the image_name column inside image_mapping table to replace "\"   with "/"
UPDATE image_mapping SET image_path = REPLACE(image_path, "\", "/");
-- SELECT image_name FROM image_mapping WHERE object_color IS NOT NULL;