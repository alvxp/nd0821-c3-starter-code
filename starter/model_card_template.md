# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

ALVXP created the model. It is a Histogram-based Gradient Boosting Classification Tree for classification purposes with maximum iteration set at 1000 and relying on the default parameters.

## Intended Use

This model should be used to predict whether a person is earning more than $50K a year.

## Training Data

The data was obtained from the Census data extracted by Barry Becker from the 1994 Census database (https://archive.ics.uci.edu/dataset/20/census+income). A set of reasonably clean records was extracted using the following conditions: ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0)).

The features are given below:

- age: continuous.
- workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
- fnlwgt: continuous.
- education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
- education-num: continuous.
- marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
- occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
- relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
- race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
- sex: Female, Male.
- capital-gain: continuous.
- capital-loss: continuous.
- hours-per-week: continuous.
- native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

To use the data for training a One Hot Encoder was used on the features and a label binarizer was used on the labels.

## Evaluation Data

After pre-processing the data, it was split into training (80%) and testing sets (20%). 

## Metrics

The metrics are given below:
- precision: 0.7755102040816326
- recall: 0.6666666666666666
- fbeta: 0.7169811320754716

## Ethical Considerations

Data bias is not expected to be significant as this was based on Census data.

## Caveats and Recommendations

A more updated Census data would be preferred given that the dataset is more than 30 years old, and may no longer be representative of the population.