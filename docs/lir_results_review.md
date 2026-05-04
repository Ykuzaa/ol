# OceanLens - Lecture des resultats LIR

Date: 2026-04-29.

Sources inspectees sur LIR, sans charger les `.npz`:

- `/scratch/emboulaalam/OceanLens_git/results/**/metrics.csv`
- `/scratch/emboulaalam/OceanLens_git/results/**/metadata.json`
- `/scratch/emboulaalam/OceanLens_git/results/**/*.png`
- `/scratch/emboulaalam/OceanLens_git/configs/variants/*.yaml`
- scripts d'inference et Slurm dans `/scratch/emboulaalam/OceanLens_git/scripts`

Les PNG ont ete copies localement ici pour inspection:

- `/homelocal/emboulaalam/OceanLens_git/tmp_lir_figures/`
- planche contact: `/homelocal/emboulaalam/OceanLens_git/tmp_lir_figures/thetao_residuals_contact_sheet.jpg`

## 1. Resultat principal

Le signal le plus clair est que le CNO `mu` est deja tres fort. Dans presque tous les runs, ajouter un FM single-member augmente le RMSE par rapport a `mu_cno`, meme si le FM ajoute des structures residuelles visibles.

Donc il ne faut pas juger le FM seulement par RMSE pixel-wise. Le RMSE favorise la moyenne conditionnelle. Le FM doit etre evalue par:

- spectres / energie haute frequence;
- coherence re-coarsening;
- CRPS et spread/skill;
- rank histograms;
- qualite des gradients/fronts;
- biais par variable et region.

En revanche, certains FM sont clairement mauvais meme visuellement et numeriquement: ils ajoutent un residu coherent a grande echelle ou des artefacts regionaux qui degradent tout.

## 2. Comparaison rapide des familles

### v3 minimal, 2019, Euler20, tile128

Dossiers:

- `results/v3_minimal_day_index_0`
- `results/v3_minimal_day_index_89`
- `results/v3_minimal_day_index_179`
- `results/v3_minimal_day_index_269`

Ces runs sont les plus propres en moyenne relative au CNO:

```text
RUN_AVG_RATIO_TO_MU
1.150 v3 day 179
1.151 v3 day 0
1.153 v3 day 89
1.159 v3 day 269
```

Interpretation:

- Le FM v3 ajoute peu de correction et reste proche de `mu`.
- Sur RMSE, il est seulement ~10-16% pire que `mu`, ce qui est acceptable pour un generatif si les spectres/details sont meilleurs.
- Le biais reste proche de zero pour `thetao` et `so`.
- Sur 2019, c'est la famille la plus stable.

Limite:

- Pas de PNG residuels disponibles dans `results` pour ces runs, seulement metrics/metadata.
- Il faut reproduire des figures v3 sur 2019 si on veut juger les structures.

### fm_variant_compare_day0, 2004-01-01, Euler20

Variants comparees:

- `v4`: baseline OT + uniform, attention.
- `v4_s1_logit_t`: OT + logit-normal, attention.
- `v4_s2_independent`: pas d'OT.
- `v4_s3_no_attn`: pas d'attention.
- `v4_s4_grad_mu`: condition `mu + grad_mu`.
- `v5_fm_only`: pas de CNO, condition LR, cible `HR - LR`.

Lecture:

- `v4_s1_logit_t` est la meilleure des variantes v4 sur ce comparatif.
- `v4` baseline produit un FM residual trop fort et biaise `thetao` positivement.
- `v4_s3_no_attn` est nettement mauvais: sans attention, gros artefacts et RMSE `thetao=1.39`.
- `v4_s4_grad_mu` n'apporte pas d'amelioration; il ajoute plutot un biais froid/negatif.
- `v5_fm_only` est mauvais pour la haute resolution physique: sans CNO, le FM apprend une correction large et bruitée, pas une bonne decomposition moyenne + residu.

Chiffres `thetao` sur 2004-01-01:

```text
mu_cno                 RMSE ~0.389
v4_s1_logit_t          RMSE 0.660
v4 restart epoch 140   RMSE 0.553
v4 baseline last       RMSE 0.887
v4_s2_independent      RMSE 0.968
v4_s4_grad_mu          RMSE 1.020
v4_s3_no_attn          RMSE 1.388
v5_fm_only             RMSE 1.469
```

Conclusion:

- Logit-normal aide.
- Attention aide.
- CNO est indispensable.
- `grad_mu` brut n'est pas une bonne condition telle quelle.
- Les checkpoints `last.ckpt` et `epoch=140.ckpt` ne racontent pas la meme histoire; il faut selectionner par validation/checkpoint, pas prendre `last` automatiquement.

### v4_s1 inference scenarios, Heun, sigma sweep, ensemble

Dossier: `results/v4_s1_inference_scenarios`

Runs:

- `heun25_sigma10_ens1`
- `heun50_sigma10_ens1`
- `heun25_sigma11_ens1`
- `heun25_sigma12_ens1`
- `heun25_sigma10_ens32`

Lecture:

- Heun25 vs Heun50 single-member ne change presque rien: `thetao RMSE 0.813` vs `0.808`.
- Augmenter `noise_sigma` degrade vite:
  - sigma 1.0: `thetao RMSE ~0.81`
  - sigma 1.1: `thetao RMSE ~0.94`
  - sigma 1.2: `thetao RMSE ~1.21`
- `ens32` est beaucoup meilleur en RMSE (`thetao RMSE 0.469`) parce que le script moyenne les membres avant metrics.

Point critique:

- `ens32` ici n'est pas une evaluation de calibration. C'est une evaluation de la moyenne d'ensemble.
- Une moyenne d'ensemble baisse naturellement la variance stochastique et le RMSE, mais peut masquer une sous/sur-dispersion.

Conclusion:

- Garder `sigma=1.0`.
- Heun25 suffit probablement pour cout/qualite.
- Evaluer les membres individuellement + CRPS/rank/spread, pas seulement la moyenne.

### v4_s1_cfg_ft

Config:

- `cond_dropout_p: 0.15`
- fine-tune 15 epochs, LR `2e-5`

Runs:

- `results/v4_s1_cfg_ft_sweep/cfg1p0_heun50_sigma10_ens1`
- `cfg1p5`
- `cfg2p0`
- `cfg3p0`
- `results/v4_s1_cfg_ft_inference/heun50_sigma10_ens32_cfg10`

Point de code important:

Dans `scripts/infer_v3_minimal.py`, si `cfg_scale == 1.0`, le script fait seulement:

```python
return fm(state, time, condition)
```

Il ne calcule pas la passe unconditional. Le vrai CFG commence donc seulement pour `cfg_scale > 1.0`:

```python
v_uncond + cfg_scale * (v_cond - v_uncond)
```

Lecture du sweep:

```text
cfg=1.0 thetao RMSE 0.691
cfg=1.5 thetao RMSE 0.942
cfg=2.0 thetao RMSE 1.307
cfg=3.0 thetao RMSE 2.500
```

Conclusion:

- Le checkpoint CFG fine-tune est utile, mais pas avec guidance forte.
- Les poids `1.5+` explosent les residus et creent un biais chaud.
- Le meilleur run CFG est en fait `cfg=1.0` ou la moyenne `ens32 cfg=1.0`.
- Il faut tester `cfg=1.05, 1.1, 1.2, 1.3`, pas sauter directement a `1.5, 2, 3`.

Le run `ens32 cfg=1.0` est le meilleur sur plusieurs variables:

```text
thetao RMSE 0.420, ratio_mu 1.08
zos    RMSE 0.0229, ratio_mu 1.11
uo     RMSE 0.0666, ratio_mu 1.02
vo     RMSE 0.0606, ratio_mu 1.02
so     RMSE 0.452, ratio_mu 1.80
```

Probleme restant: `so` reste nettement degradee par rapport a `mu`.

### v4_s1_recoarsen_ft

Config:

- fine-tune S1 avec `recoarsen.weight: 0.05`
- warmup 3 epochs
- `max_epochs: 10`

Run: `results/v4_s1_recoarsen_ft_inference/heun50_sigma10_ens1`

Lecture:

- `thetao RMSE 0.756`, meilleur que S1 Heun single (`0.808`) mais moins bon que CFG ens32 ou v3.
- `so RMSE 1.304`, tres degrade.
- `zos/u/v` legerement meilleurs que certains single runs, mais restent moins bons que `mu`.

Interpretation:

- La contrainte re-coarsening telle quelle ne suffit pas.
- Elle peut pousser certaines variables, surtout salinite, dans une mauvaise direction.
- Il faut logger la conservation re-coarsen directement, parce que le RMSE seul ne dit pas si cette variante remplit son objectif.

Conclusion:

- Ne pas jeter l'idee, mais revoir le poids par variable.
- `so` ne doit probablement pas recevoir le meme comportement que `thetao`.
- Tester re-coarsen sur `thetao` seul ou poids variables plus prudents.

### v4_s1_regional_ft

Run: `results/v4_s1_regional_ft_inference/heun50_sigma10_ens1`

Lecture:

- Degradation nette: `thetao RMSE 0.889`, `so RMSE 1.536`.
- La fine-tune regionale n'ameliore pas le cas global 2004-01-01.

Interpretation:

- Possible specialisation locale au detriment global.
- Possible distribution shift des patchs regionaux.
- Peut rester utile sur Med/Black/Arctic, mais il faut evaluer sur ces regions separement.

Conclusion:

- Ne pas utiliser cette variante comme modele global.
- Si on veut du regional, produire des metrics region-specific, pas global mean.

### v4_s2 independent + Heun ablations

Dossier: `results/v4_s2_ablation_solvers`

Lecture:

- `v4_s2_independent` reste mauvais.
- Heun25 vs Heun50 ne sauve pas le modele.
- Sigma 1.1/1.2 degrade encore plus.

Conclusion:

- Le probleme n'est pas le solveur; c'est le training/coupling/condition.
- Garder OT minibatch pour les meilleures variantes.

## 3. Classement utile par variable

Meilleurs RMSE pred observés:

```text
thetao:
  0.420 v4_s1_cfg_ft ens32 cfg=1.0
  0.449 v3 day 89
  0.461 v3 day 269
  0.469 v4_s1 ens32

so:
  0.297 v3 day 89
  0.300 v3 day 0
  0.339 v3 day 269
  0.369 v3 day 179
  0.452 v4_s1_cfg_ft ens32

zos:
  0.0229 v4_s1_cfg_ft ens32
  0.0243 v4_s1 ens32
  0.0255 v3 day 179

uo:
  0.0666 v4_s1_cfg_ft ens32
  0.0667 v4_s1 ens32

vo:
  0.0606 v4_s1_cfg_ft ens32
  0.0607 v4_s1 ens32
```

Lecture:

- Pour `thetao`, `zos`, `uo`, `vo`, la moyenne d'ensemble v4_s1/CFG est tres competitive.
- Pour `so`, v3 reste beaucoup meilleur.
- La salinite est le canal le plus problematique pour les FM v4/v4_s1 actuels.

## 4. Lecture visuelle des PNG thetao

Les figures montrent trois motifs:

1. Le CNO residual ressemble fortement a `HR - LR` en grandes structures, avec des fronts et une texture coherente.
2. Plusieurs FM residuals introduisent des nappes larges positives/negatives, pas seulement des details fins.
3. Les mauvais runs ont des artefacts regionaux forts:
   - v4 baseline: residus chauds forts pres des cotes et bassins.
   - no-attn: larges zones froides/chaudes incoherentes.
   - sigma eleve: amplitude trop forte.
   - CFG fort: biais chaud et structures trop amplifiees.

Le meilleur visuel relatif est le run `v4_s1_cfg_ft_inference/heun50_sigma10_ens32_cfg10`: le FM residual est plus doux, mais il ressemble aussi a une correction moyenne faible. C'est bon pour RMSE, pas encore une preuve de bonne stochasticite.

## 5. Conclusions d'architecture

### Ce qui marche

- CNO residual/log-gradient est le socle le plus fiable.
- Decomposition `HR = mu + FM_residual` est meilleure que FM-only.
- Logit-normal est meilleur que uniform dans les variantes v4.
- Attention est necessaire.
- OT minibatch semble meilleur que independent.
- Ensemble averaging stabilise fortement les predictions.

### Ce qui ne marche pas encore

- CFG fort (`cfg>=1.5`) explose le residu.
- Noise inflation simple (`sigma=1.1/1.2`) degrade.
- FM-only est trop faible.
- `grad_mu` concatene brut n'aide pas.
- Recoarsen fine-tune global degrade beaucoup `so`.
- Regional fine-tune degrade le global.

### Le risque principal

Le FM apprend souvent une correction basse frequence ou spatialement biaisee, alors qu'on veut surtout un residu haute frequence physiquement compatible avec LR.

Autrement dit: il ne faut pas seulement "augmenter la variance"; il faut controler ou cette variance apparait.

## 6. Prochaines experiences recommandees

1. Reproduire v3 sur 2004-01-01
   - Meme jour que les v4.
   - Meme tile size si possible.
   - Generer PNG thetao.
   - But: separer effet split/checkpoint/architecture.

2. Evaluer les ensembles comme ensembles
   - Ne pas seulement sauver la moyenne.
   - Sauver metrics par membre ou calculer CRPS/spread/rank.
   - Garder `ens32 mean` comme diagnostic secondaire.

3. CFG fin
   - Tester `cfg_scale = 1.0, 1.05, 1.1, 1.2, 1.3`.
   - Arreter `1.5+` pour l'instant.
   - Mesurer par variable, surtout `so`.

4. Recoarsen par variable
   - Tester `thetao` seul.
   - Tester poids plus faibles pour `so`.
   - Logger erreur `coarsen(pred)-LR` directement.

5. Filtrer la cible FM
   - Entrainer FM sur residu haute frequence seulement:
     `target = HR - mu`, puis retirer une composante basse frequence par pooling.
   - Ou ajouter une penalite basse frequence pour que le FM ne change pas le bilan.

6. Diagnostics spectraux
   - Comparer `HR`, `LR_up`, `mu`, `pred`, `FM_residual`.
   - Une bonne variante doit ajouter energie aux bonnes echelles sans injecter du biais bassin.

7. Selection de checkpoint
   - Ne pas utiliser `last.ckpt` par defaut.
   - Le `v4-fm-epoch=140.ckpt` est meilleur que certains `last.ckpt`.
   - Sauver et comparer top-k par validation physique, pas seulement loss.

