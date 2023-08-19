from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from matplotlib.gridspec import GridSpec
from matplotlib import cm, colors
import pickle

df = pd.read_csv('/home/ryley/WDK/ML/dataset/turbulence_dataset_clean.csv')

#input_features = ['komega_I1_1','komega_I1_3','komega_I1_5','komega_q2']#'komega_I1_2']#,'komega_I1_2','komega_I1_3','komega_I1_5','komega_q2']
input_features = [
#'komegasst_I1_1','komegasst_I1_3','komegasst_I1_5','komegasst_q5','komegasst_q6']
'komegasst_I1_1','komegasst_I1_3','komegasst_I1_5','komegasst_q5','komegasst_q6',]#'komegasst_q9'] #not bad
#'komegasst_q5','komegasst_I1_1','komegasst_I1_3','komegasst_I2_3','komegasst_I1_5']
#'komegasst_q5','komegasst_q6','komegasst_I1_1','komegasst_I1_3','komegasst_I2_3','komegasst_I1_4','komegasst_I2_4','komegasst_I1_5']
#'komegasst_q6','komegasst_q5','komegasst_I1_1','komegasst_I2_1','komegasst_I1_3','komegasst_I1_4','komegasst_I2_4']#,'komegasst_q5','komegasst_I1_1','komegasst_I2_1','komegasst_I1_3','komegasst_I1_21']#,'komegasst_q6','komegasst_I1_1','komegasst_I1_3','komegasst_I2_3','komegasst_I1_4','komegasst_I2_4','komegasst_I1_5']
#'komegasst_q6','komegasst_I1_1','komegasst_I2_1','komegasst_I2_7','komegasst_I1_3','komegasst_I1_5','komegasst_I1_9','komegasst_I2_4','komegasst_I1_4','komegasst_I1_10','komegasst_I1_16','komegasst_I1_21']#'komegasst_I1_1','komegasst_I1_3','komegasst_I1_4','komegasst_I1_16']
#'komegasst_q5','komegasst_I2_1','komegasst_I1_4','komegasst_I2_3','komegasst_I1_9','komegasst_I1_25','komegasst_I1_16','komegasst_I1_21','komegasst_I1_17']
#'komegasst_q5','komegasst_I2_3','komegasst_I2_6','komegasst_I1_4','komegasst_I2_5','komegasst_I1_16',]
#'komegasst_I1_16','komegasst_I2_5','komegasst_I1_1','komegasst_I2_4','komegasst_I1_21']

cases = ['case_0p5','case_0p8','case_1p5','fp_1410','fp_2540','fp_4060','squareDuctAve_Re_1100','squareDuctAve_Re_2000','squareDuctAve_Re_3500']
#cases = ['case_0p5','case_0p8','case_1p5','fp_1000', 'fp_1410', 'fp_2540', 'fp_3030', 'fp_3270','fp_4060']
#cases = ['case_0p5','case_0p8','fp_2540','squareDuctQuad1_Re_2000']

n_comp = 4
X = np.empty((len(cases)*100, len(input_features)))
for i, case in enumerate(cases):
    df_i = df[df.Case == case][input_features]
    X[i*100:(i+1)*100,:] = df_i.sample(n = 100,random_state=0,replace=False).to_numpy()

print(X[0:5])
#X = df[df.Case.isin(cases)][input_features]
scaler = StandardScaler()
X = scaler.fit_transform(X)
model = GaussianMixture(n_components=n_comp, random_state=0,tol=1E-10,max_iter=1000)

model.fit(X)
pickle.dump(model,open('models/splitr/splitr.pkl', 'wb'))
pickle.dump(scaler,open('models/splitr/splitr_scaler.pkl','wb'))
print(f'Model iterations: {model.n_iter_}')

fig, axs = plt.subplots(4,1, figsize=(12,20))

for i, case in enumerate(['fp_1410','fp_2540','fp_4060']):
    df_case = scaler.transform(df[df.Case == case][input_features])
    yplus = np.load(f'/home/ryley/WDK/ML/dataset/numpy/komegasst/komegasst_{case}_yplus.npy')
    Uplus = np.load(f'/home/ryley/WDK/ML/dataset/numpy/komegasst/komegasst_{case}_Uplus.npy')[:,0]

    y = model.predict(scaler.transform(df[df.Case==case][input_features]))
    clusters = np.unique(y)
    print(clusters)
    norm = colors.Normalize(0, n_comp)
    cmap = cm.get_cmap("Set1")
    axs[i].scatter(yplus,Uplus,c=cmap(norm(y)))
    axs2 = axs[i].twinx()
    for i,feature in enumerate(input_features):
        axs2.scatter(yplus,df_case[:,i],label=feature,s=5)
    axs2.legend()
    axs2.set_ylim([-2,3])
    
    for ax in axs[0:-1]:
        ax.semilogx()
        ax.set_ylim([0,27])
        ax.set_xlim([5E-2,5E3])

        ax.plot([10,10],[0,30],'k--')
        ax.plot([30,30],[0,30],'k--')
        
fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),orientation='horizontal',pad=0.2, ax=axs[-1])
fig.savefig('models/splitr/splitr_fp.png',dpi=300)
fig, axs = plt.subplots(3,1, figsize=(7,10))
for i, case in enumerate(['case_0p5','case_1p0','case_1p5']):
    y = model.predict(scaler.transform(df[df.Case==case][input_features]))
    clusters = np.unique(y)
    print(clusters)
    norm = colors.Normalize(0, n_comp)
    cmap = cm.get_cmap("Set1")

    Cx = df[df.Case == case]['komegasst_C_1']
    Cy = df[df.Case == case]['komegasst_C_2']
    axs[i].scatter(Cx.values, Cy.values,s=1.5,c=cmap(norm(y)))
    for ax in axs:
        ax.set_aspect(1)
fig.savefig('models/splitr/splitr_phll.png',dpi=300)
fig, axs = plt.subplots(3,1, figsize=(7,10))
for i, case in enumerate(['squareDuctAve_Re_1100','squareDuctAve_Re_2000','squareDuctAve_Re_3500']):
    y = model.predict(scaler.transform(df[df.Case==case][input_features]))
    clusters = np.unique(y)
    print(clusters)
    norm = colors.Normalize(0, n_comp)
    cmap = cm.get_cmap("Set1")
    Cy = df[df.Case == case]['komegasst_C_2']
    Cz = df[df.Case == case]['komegasst_C_3']
    axs[i].scatter(Cy.values, Cz.values,s=4,c=cmap(norm(y)))
    for ax in axs:
        ax.set_aspect(1)

fig.savefig('models/splitr/splitr_duct.png',dpi=300)


fig = plt.figure(layout="tight",figsize=(20,60))

gs = GridSpec(5*n_comp, 2, figure=fig)

for i, case in enumerate(['case_0p5','case_0p8','case_1p5']):
    y = model.predict(scaler.transform(df[df.Case==case][input_features]))
    probab = model.predict_proba(scaler.transform(df[df.Case==case][input_features]))
    print(y)
    clusters = np.unique(y)
    print(clusters)
    Cx = df[df.Case == case]['komegasst_C_1']
    Cy = df[df.Case == case]['komegasst_C_2']
    ax_clusters = fig.add_subplot(gs[i*n_comp:(i+1)*n_comp,0])
    ax_clusters.scatter(Cx.values, Cy.values,s=1.5,c=cmap(norm(y)))
    ax_clusters.set_aspect(1)
    for cluster in clusters:    
        row_ix = np.where(y == cluster)
        #ax_clusters.scatter(Cx.values[row_ix], Cy.values[row_ix],s=3.0)
        ax_probab = fig.add_subplot(gs[(i*n_comp+cluster),1])
        ax_probab.scatter(Cx.values, Cy.values,s=0.2, c=probab[:,cluster],cmap='Blues')
        ax_probab.set_aspect(1)
    # create scatter of these samples

y = model.predict(scaler.transform(df[input_features]))
print(np.unique(y, return_counts=True))
fig.savefig('models/splitr/splitr_probab.png',dpi=300)


df['Cluster'] = model.predict(scaler.transform(df[input_features]))
probs = model.predict_proba(scaler.transform(df[input_features]))

for i in range(n_comp):
    df[f'Probability_cluster_{i}'] = probs[:,i]

df.to_csv('/home/ryley/WDK/ML/dataset/turbulence_dataset_clean_split.csv')