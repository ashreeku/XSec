for config in pdf phishing netflow bodmas nsl_kdd_multi
do
    # comparison with baselines
    python fidelity_xsec.py --config "../config/${config}.json"
    python fidelity_lime.py --config "../config/${config}.json"
    python fidelity_shap.py --config "../config/${config}.json"
    python fidelity_ig.py --config "../config/${config}.json"
    python fidelity_ggc.py --config "../config/${config}.json"

    # hyperparameter sensitivity
    for k in 20 30
    do
        python fidelity_xsec.py --config "../config/${config}.json" --num_prototypes_per_class ${k}
    done

    for n in 1 5 7 10
    do
        python fidelity_xsec.py --config "../config/${config}.json" --num_similarity_scores ${n}
    done
    
    # ablation study
    for ablation in binary sparsity diversity cluster crossentropy
    do
        python fidelity_xsec.py --config "../config/${config}.json" --ablation ${ablation}
    done
done

# evaluate lemna separately since it is only for binary datasets
for config in pdf phishing netflow
do
    python fidelity_lemna.py --config "../config/${config}.json"
done

python plot_fidelity_baseline_comparison.py
python plot_fidelity_hyperparameter_sensitivity_k.py
python plot_fidelity_hyperparameter_sensitivity_n.py
python plot_fidelity_ablation_study.py