# ============================================
# NOTEBOOK 02 : PRÉTRAITEMENT MULTI-OMIQUE
# ============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
import joblib
import warnings
warnings.filterwarnings('ignore')

# Chemins (Windows absolu)
PROJECT_ROOT = Path(r'C:\Users\nidha\Desktop\xai_clinical_prediction')
DATA_DIR = PROJECT_ROOT / 'data' / 'raw' / 'brca_tcga'
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
FIGURES_DIR = PROJECT_ROOT / 'figures'

for d in [PROCESSED_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("="*70)
print("PRÉTRAITEMENT MULTI-OMIQUE COMPLET")
print("="*70)
print(f"DATA_DIR: {DATA_DIR}")
print(f"Existe: {DATA_DIR.exists()}")
print("\n[INPUT] Chargement des données brutes")

# Fonction pour trouver les fichiers .txt
def load_tsv_file(directory, basename):
    """Charge un fichier TSV avec extension .txt"""
    filepath = directory / f"{basename}.txt"
    if not filepath.exists():
        raise FileNotFoundError(f"Fichier non trouvé: {filepath}")
    
    # Détecter les lignes de commentaires (#)
    with open(filepath, 'r') as f:
        first_lines = [f.readline() for _ in range(5)]
    
    skip_rows = sum(1 for line in first_lines if line.startswith('#'))
    
    return pd.read_csv(filepath, sep='\t', skiprows=skip_rows)

# 1. Données cliniques
print("\n[1/5] Chargement clinique...")
try:
    clinical = load_tsv_file(DATA_DIR, 'data_clinical_patient')
    print(f"[SUCCESS] Clinical: {clinical.shape}")
except Exception as e:
    print(f"[ERROR] Clinical error: {e}")
    exit(1)

# Sauvegarder pour réutilisation
clinical.to_csv(PROCESSED_DIR / 'clinical_raw.csv', index=False)

# 2. mRNA RSEM
print("\n[2/5] Chargement mRNA RSEM...")
try:
    mrna = load_tsv_file(DATA_DIR, 'data_mrna_seq_v2_rsem')
    print(f"[SUCCESS] mRNA: {mrna.shape}")
    mrna_t = mrna.set_index('Hugo_Symbol').drop('Entrez_Gene_Id', axis=1, errors='ignore').T
    mrna_t.index = [s[:12] for s in mrna_t.index]
    mrna_t = mrna_t[~mrna_t.index.duplicated(keep='first')]
except Exception as e:
    print(f"[ERROR] mRNA error: {e}")
    exit(1)

# 3. Méthylation HM450 (optionnel)
print("\n[3/5] Chargement méthylation...")
try:
    meth = load_tsv_file(DATA_DIR, 'data_methylation_hm450')
    possible_index_cols = ['Composite Element REF', 'Hybridization REF', 'ID_REF', 'Probe ID', 'IlmnID']
    index_col = next((col for col in possible_index_cols if col in meth.columns), meth.columns[0])
    meth_t = meth.set_index(index_col).T
    meth_t.index = [s[:12] for s in meth_t.index]
    meth_t = meth_t[~meth_t.index.duplicated(keep='first')]
    print(f"[SUCCESS] Methylation: {meth_t.shape}")
    has_methylation = True
except Exception as e:
    print(f"[WARNING] Méthylation non disponible: {e}")
    has_methylation = False

# 4. CNA (optionnel)
print("\n[4/5] Chargement CNA...")
try:
    cna = load_tsv_file(DATA_DIR, 'data_cna')
    cna_t = cna.set_index('Hugo_Symbol').T
    cna_t.index = [s[:12] for s in cna_t.index]
    cna_t = cna_t[~cna_t.index.duplicated(keep='first')]
    print(f"[SUCCESS] CNA: {cna_t.shape}")
    has_cna = True
except Exception:
    print("[WARNING] CNA non disponible")
    has_cna = False

# 5. Mutations (optionnel)
print("\n[5/5] Chargement mutations...")
try:
    mut = load_tsv_file(DATA_DIR, 'data_mutations')
    print(f"[SUCCESS] Mutations: {mut.shape}")
    has_mutations = True
except Exception:
    print("[WARNING] Mutations non disponible")
    has_mutations = False

print("\n[TARGET] Création de la cible (Survie Globale à 5 ans)")

def create_target(clinical_df, cutoff_months=60):
    df = clinical_df.copy()
    df.index = df['PATIENT_ID'].astype(str).str[:12].str.upper()
    time = pd.to_numeric(df['OS_MONTHS'], errors='coerce')
    event = df['OS_STATUS'].astype(str)
    event_binary = event.str.contains('DECEASED|Dead|1', case=False, na=False).astype(int)
    target = pd.Series(float('nan'), index=df.index)
    target.loc[time > cutoff_months] = 0
    target.loc[(event_binary == 1) & (time <= cutoff_months)] = 1
    target = target.groupby(level=0).max().dropna()
    return target

y_series = create_target(clinical)
y_df = pd.DataFrame({'PATIENT_ID_SHORT': y_series.index, 'TARGET': y_series.values})
clinical['PATIENT_ID_SHORT'] = clinical['PATIENT_ID'].astype(str).str[:12].str.upper()
clinical = clinical.merge(y_df, on='PATIENT_ID_SHORT', how='inner')
y = clinical['TARGET'].astype(int)

print(f"Distribution cible: {y.value_counts().to_dict()}")

# Features cliniques
print("\n[FEATURES] Features cliniques")
clinical_features = ['AGE', 'AJCC_PATHOLOGIC_TUMOR_STAGE', 'HISTOLOGICAL_DIAGNOSIS', 
                     'ER_STATUS_BY_IHC', 'PR_STATUS_BY_IHC', 'IHC_HER2']
X_clinical = clinical[clinical_features].copy()

# Fix AGE
age_numeric = pd.to_numeric(X_clinical['AGE'], errors='coerce')
X_clinical['AGE'] = age_numeric.fillna(age_numeric.median())

stage_map = {'Stage I': 1, 'Stage IA': 1, 'Stage IB': 1, 'Stage II': 2, 'Stage IIA': 2, 'Stage IIB': 2,
             'Stage III': 3, 'Stage IIIA': 3, 'Stage IIIB': 3, 'Stage IIIC': 3, 'Stage IV': 4}
X_clinical['TUMOR_STAGE_ENC'] = X_clinical['AJCC_PATHOLOGIC_TUMOR_STAGE'].map(stage_map).fillna(0)
X_clinical['GRADE_ENC'] = X_clinical['HISTOLOGICAL_DIAGNOSIS'].astype(str).str.extract(r'Grade (\d)')[0].fillna(0).astype(int)
status_map = {'Positive': 1, 'Negative': 0, 'Indeterminate': 0.5, 'Equivocal': 0.5}
for col in ['ER_STATUS_BY_IHC', 'PR_STATUS_BY_IHC', 'IHC_HER2']:
    X_clinical[col] = X_clinical[col].map(status_map).fillna(0)

X_clinical = X_clinical[['AGE', 'TUMOR_STAGE_ENC', 'GRADE_ENC', 'ER_STATUS_BY_IHC', 'PR_STATUS_BY_IHC', 'IHC_HER2']]

# Features génomiques
print("\n[FEATURES] Features génomiques")
gene_vars = mrna_t.var().sort_values(ascending=False)
top_genes = gene_vars.head(200).index
X_mrna = np.log2(mrna_t[top_genes] + 1)
X_mrna.columns = [f'GENE_{c}' for c in X_mrna.columns]

# Fusion multi-omique
print("\n[FUSION] Fusion multi-omique")
clinical_ids = clinical['PATIENT_ID_SHORT'].values
all_ids = [set(clinical_ids), set(mrna_t.index)]
if has_methylation: all_ids.append(set(meth_t.index))
if has_cna: all_ids.append(set(cna_t.index))
common_ids = sorted(set.intersection(*all_ids))
print(f"Échantillons communs: {len(common_ids)}")

# Re-align everything to common_ids
X_clinical_f = X_clinical.set_index(clinical['PATIENT_ID_SHORT']).loc[common_ids].reset_index(drop=True)
y_f = y.set_axis(clinical['PATIENT_ID_SHORT']).loc[common_ids].reset_index(drop=True)
X_mrna_f = X_mrna.loc[common_ids].reset_index(drop=True)

features_list = [X_clinical_f, X_mrna_f]
if has_methylation:
    X_meth_f = meth_t.loc[common_ids].reset_index(drop=True)
    meth_vars = X_meth_f.var().sort_values(ascending=False)
    X_meth_f = X_meth_f[meth_vars.head(100).index]
    X_meth_f.columns = [f'METH_{c}' for c in X_meth_f.columns]
    features_list.append(X_meth_f)

X_combined = pd.concat(features_list, axis=1)
print(f"Features combinées: {X_combined.shape}")

# Split
print("\n[SPLIT] Split train/val/test")
X_temp, X_test, y_temp, y_test = train_test_split(X_combined, y_f, test_size=0.2, random_state=42, stratify=y_f)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp)

# Imputation (sur TRAIN, appliqué à Val/Test)
print("\n[IMPUTE] Imputation des valeurs manquantes")
imputer = SimpleImputer(strategy='median')
X_train_imp = imputer.fit_transform(X_train)
X_val_imp = imputer.transform(X_val)
X_test_imp = imputer.transform(X_test)

# Convertir en DataFrame pour garder les noms de colonnes pour SelectKBest support
X_train_imp_df = pd.DataFrame(X_train_imp, columns=X_train.columns)

# Selection
print("\n[SELECTION] Feature selection")
selector = SelectKBest(f_classif, k=min(150, X_train.shape[1]))
X_train_sel = selector.fit_transform(X_train_imp_df, y_train)
selected_features = X_train.columns[selector.get_support()]
X_val_sel = selector.transform(X_val_imp)
X_test_sel = selector.transform(X_test_imp)

# Scaling
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_sel)
X_val_sc = scaler.transform(X_val_sel)
X_test_sc = scaler.transform(X_test_sel)

# Sauvegarde
print("\n[SAVE] Sauvegarde des données")
np.save(PROCESSED_DIR / 'X_train.npy', X_train_sc)
np.save(PROCESSED_DIR / 'X_val.npy', X_val_sc)
np.save(PROCESSED_DIR / 'X_test.npy', X_test_sc)
np.save(PROCESSED_DIR / 'y_train.npy', y_train)
np.save(PROCESSED_DIR / 'y_val.npy', y_val)
np.save(PROCESSED_DIR / 'y_test.npy', y_test)

joblib.dump(scaler, PROCESSED_DIR / 'scaler.joblib')
joblib.dump(list(selected_features), PROCESSED_DIR / 'selected_features.joblib')

print("\n[SUCCESS] PRÉTRAITEMENT TERMINÉ!")