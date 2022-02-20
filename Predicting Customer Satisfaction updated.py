# changed the font size
fig = plt.figure(figsize = (20,8))
ax = plt.axes() 
sns.barplot(x = df_ecommerce.product_category.value_counts().index[:10], 
            y = df_ecommerce.product_category.value_counts()[:10], ax = ax)
sns.set(font_scale = 1)
ax.set_xlabel('Product category', fontsize = 25)
ax.set_ylabel('The quantity of order', fontsize = 25)
fig.suptitle("Top 10 best purchased product by customers", fontsize = 25)
plt.show()

# Measure accuracy of the LinearRegression model using K-fold cross validation
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

cross_val_score(LinearRegression(), X, y, cv=cv)

# Finding best model and its best parameter
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

def find_best_model_using_gridsearchcv(X, y):
    algos = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                'normalize': [True, False],
                'fit_intercept': [True, False]
            }
        },

        'gradient_boosting': {
            'model': GradientBoostingClassifier(),
            'params': {
                'n_estimators': (50, 100),
                'max_depth':(4,5)
            }
        },

         'random_forest' : {
            'model' : RandomForestClassifier(),
            'params': {
                'n_estimators': (10,20),
                'max_depth':(1,5)
            }

        },'decision_tree_classifier':{
            'model': DecisionTreeClassifier(),
            'params': {
                'max_depth':  [1, 5],
                'min_samples_split': [5, 100]
            }
        },

        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random', 'cyclic']
            }
        },
      
        'decision_tree_regressor': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['mse','friedman_mse'],
                'splitter': ['best','random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    for algo_name, config in algos.items():
        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X, y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

find_best_model_using_gridsearchcv(X, y)
