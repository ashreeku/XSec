for config in pdf phishing netflow bodmas nsl_kdd_multi
do
    # comparison with baselines
    python run_xsec.py --config ../config/${config}.json
    python run_xsec.py --config ../config/${config}.json --train
    python run_lime.py --config ../config/${config}.json
    python run_lime.py --config ../config/${config}.json --train
    python run_shap.py --config ../config/${config}.json
    python run_shap.py --config ../config/${config}.json --train
    python run_ig.py --config ../config/${config}.json
    python run_ig.py --config ../config/${config}.json --train
    python run_ggc.py --config ../config/${config}.json
    python run_ggc.py --config ../config/${config}.json --train
    python lemna_save_data_for_R.py --config ../config/${config}.json

    # hyperparameter sensitivity
    for k in 20 30
    do
        python run_xsec.py --config ../config/${config}.json --num_prototypes_per_class ${k}
    done

    for n in 1 5 7 10
    do
        python run_xsec.py --config ../config/${config}.json --num_similarity_scores ${n}
    done

    # ablation study
    for ablation in binary sparsity diversity cluster crossentropy
    do
        python run_xsec.py --config ../config/${config}.json --ablation ${ablation}
    done
done

# scalability
python run_xsec.py --config ../config/bodmas200.json --num_prototypes_per_class 20
python run_xsec.py --config ../config/bodmas500.json --num_prototypes_per_class 50

# generate explanations for lemna
Rscript lemna_generate_explanations.r 3317 pdf TRUE
Rscript lemna_generate_explanations.r 5311 phishing TRUE
Rscript lemna_generate_explanations.r 6558 netflow TRUE

# save importance scores for lemna
for config in pdf phishing netflow
do
    python run_lemna.py --config ../config/${config}.json
    python run_lemna.py --config ../config/${config}.json --train
done

# generate importance scores for test data with alternate seed to compute stability
for config in pdf phishing netflow bodmas nsl_kdd_multi
do
    python run_xsec.py --config ../config/${config}.json --alternate
    python run_lime.py --config ../config/${config}.json --alternate
    python run_shap.py --config ../config/${config}.json --alternate
    python run_ig.py --config ../config/${config}.json --alternate
    python run_ggc.py --config ../config/${config}.json --alternate
done

Rscript lemna_generate_explanations.r 7705 pdf FALSE
Rscript lemna_generate_explanations.r 7982 phishing FALSE
Rscript lemna_generate_explanations.r 8375 netflow FALSE

for config in pdf phishing netflow
do
    python run_lemna.py --config ../config/${config}.json --alternate
done