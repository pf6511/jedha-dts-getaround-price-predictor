DO $$
DECLARE
    run_id_col text;
    exp_id int;
BEGIN
    -- 0. Find experiment_id from experiment name
    SELECT experiment_id
    INTO exp_id
    FROM experiments
    WHERE name = 'getaround_rental_price_predictor';

    IF exp_id IS NULL THEN
        RAISE EXCEPTION 'No experiment found with that name.';
    END IF;

    -- 1. Detect run_id or run_uuid
    SELECT column_name
    INTO run_id_col
    FROM information_schema.columns
    WHERE table_name = 'runs'
      AND column_name IN ('run_id', 'run_uuid')
    LIMIT 1;

    IF run_id_col IS NULL THEN
        RAISE EXCEPTION 'No run_id or run_uuid column found in runs table.';
    END IF;

    RAISE NOTICE 'Using column: %', run_id_col;
    RAISE NOTICE 'Deleting experiment_id: %', exp_id;

    -- 2. Delete metrics
    EXECUTE format(
        'DELETE FROM metrics
         WHERE %I IN (SELECT %I FROM runs WHERE experiment_id = $1)',
        run_id_col, run_id_col
    ) USING exp_id;

    -- 3. Delete params
    EXECUTE format(
        'DELETE FROM params
         WHERE %I IN (SELECT %I FROM runs WHERE experiment_id = $1)',
        run_id_col, run_id_col
    ) USING exp_id;

    -- 4. Delete tags
    EXECUTE format(
        'DELETE FROM tags
         WHERE %I IN (SELECT %I FROM runs WHERE experiment_id = $1)',
        run_id_col, run_id_col
    ) USING exp_id;

    -- 5. Delete latest_metrics
    EXECUTE format(
        'DELETE FROM latest_metrics
         WHERE %I IN (SELECT %I FROM runs WHERE experiment_id = $1)',
        run_id_col, run_id_col
    ) USING exp_id;

    -- 6. Delete from logged_model_* tables
    DELETE FROM logged_model_params WHERE experiment_id = exp_id;
    DELETE FROM logged_model_tags WHERE experiment_id = exp_id;
    DELETE FROM logged_model_metrics WHERE experiment_id = exp_id;

    -- 7. Delete runs
    EXECUTE format(
        'DELETE FROM runs WHERE experiment_id = $1',
        run_id_col
    ) USING exp_id;

    -- 8. Delete experiment
    DELETE FROM experiments WHERE experiment_id = exp_id;

    RAISE NOTICE 'Experiment "%" (ID=%) deleted successfully', 'getaround_rental_price_predictor', exp_id;
END$$;
