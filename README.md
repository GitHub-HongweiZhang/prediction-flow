# prediction-flow
**prediction-flow** is a Python package providing modern **Deep-Learning**
based CTR models. Models are implemented by **PyTorch**.

## how to use
* Clone the code.
```
git clone https://github.com/GitHub-HongweiZhang/prediction-flow.git
```
* install
```
pip install -e .
```

## feature
### how to define feature
There are two parameters for all feature types, name and column_flow.
The name parameter is used to index the column raw data from input data frame.
The column_flow parameter is a single transformer of a list of transformers.
The transformer is used to pre-process the column data before training the model.

* dense number  feature
```
Number('age', StandardScaler())
Number('ctr', None)
```
* sparse category feature
```
Category('movieId', CategoryEncoder(min_cnt=1))
```
* var length sequence feature
```
Sequence('genres', SequenceEncoder(sep='|', min_cnt=1))
```

## transformer
The following transformers are provided now.
| transformer | supported feature type | detail |
|--|--|--|
| StandardScaler | Number | Wrapper of scikit-learn's StandardScaler. Null value must be filled in advance. |
| LogTransformer | Number | Log scaler. Null value must be filled in advance. |
| CategoryEncoder | Category | Converting str value to int. Null value must be filled in advance using '\_\_UNKNOWN\_\_'. |
| SequenceEncoder | Sequence | Converting sequence str value to int. Null value must be filled in advance using '\_\_UNKNOWN\_\_'. |

## model
| model | reference |
|--|--|
| DNN | - |
| DeepFM | [IJCAI 2017][DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](http://www.ijcai.org/proceedings/2017/0239.pdf) |
| DIN | [KDD 2018][Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1706.06978.pdf) |
| OTHER | TODO |

## example
### movielens-1M
* Prepare the dataset. [preprocess.ipynb](https://github.com/GitHub-HongweiZhang/prediction-flow/blob/master/examples/movielens/ml-1m/preprocess.ipynb)
* Run the model. [movielens-1m.ipynb](https://github.com/GitHub-HongweiZhang/prediction-flow/blob/master/examples/movielens/movielens-1m.ipynb)

## acknowledge and reference
* Referring the design from [DeepCTR](https://github.com/shenweichen/DeepCTR),
  the features are divided into dense (class Number), sparse (class Category),
  sequence (class Sequence) types.
