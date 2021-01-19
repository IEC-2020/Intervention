##############################################################This code will generate correlation between the features used in the dataset###########################

#Generation correlation between the variables in FGF21#
X_variable = pd.DataFrame(data=X).  ############################# X is your preprocessed input dataset ###################
corr = X_variable.corr(method='pearson')
fig = plt.figure(figsize=(50,45))
ax = fig.add_subplot(111)
#cax = ax.matshow(corr,annot=True, cmap='coolwarm', vmin=-1, vmax=1) ################### cmap use to change the color of the heatmap #############################
cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(X_variable.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(X_variable.columns)
ax.set_yticklabels(X_variable.columns)
plt.show() 
corr.to_excel('corr.xlsx')
