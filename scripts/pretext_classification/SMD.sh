python exp_train.py --exp-name "pretext_classification with PLAD" \
                    --task pretext_classification \
                    --data SMD \
                    --positive-augmentor-time 1218_0646 \
                    --perturbator-time 0105_2008 \
                    --deviation-mode abs \
                    --non-constant-dim-tau 0.5 \
                    --constant-dim-tau 0.5 \
                    --perturbator-mode plad \
                    --seed 1