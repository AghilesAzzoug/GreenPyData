<p align="center">
  <!-- Unit Tests -->
  <a href="https://github.com/AghilesAzzoug/GreenPyData/actions/workflows/build-and-test.yml">
    <img src="https://github.com/AghilesAzzoug/GreenPyData/actions/workflows/build-and-test.yml/badge.svg" alt="tests">
  </a>
  <!-- License -->
  <a href="https://opensource.org/license/gpl-3-0/">
        <img src="https://img.shields.io/github/license/AghilesAzzoug/GreenPyData" alt="gpl_license">
  </a>
</p>

# GreenPyData Plugin for (PyTorch) Data Scientists

As data science continues to grow in popularity (see LLMs...), it is becoming increasingly important to consider the
environmental impact of the code we write.
Many data science tasks, especially deep learning ones, require significant computational resources, which in turn
generate carbon emissions and contribute to climate change.

It is highly inspired from https://github.com/green-code-initiative/ecoCode which is a project really worth checking!
And using!

_GreenPyData_ is a humble try from a data scientist who is interested in sustainability and eco-friendliness in 
software development and data science.

## Introduction

_GreenPyData_ is an open-source SonarQube plugin designed specifically for data scientists (who use PyTorch). Its purpose is
to assist in eco-designing your code by identifying and flagging energy-intensive or computationally inefficient code
segments that can be optimized to reduce carbon footprint and improve performance.

**Currently, _GreenPyData_ only supports PyTorch. However, we plan to support other frameworks in the future.**

To install _GreenPyData_, follow these steps:

* Install SonarQube on your system (version 7.9 or higher).
* Download the _GreenPyData_ plugin from our GitHub repository.
* Create the .jar (`mvn clean package -DskipTests`) and copy it into the extensions/plugins directory of your SonarQube
  installation.
* Restart SonarQube.

## Usage

Once _GreenPyData_ is installed, you can use it to analyze your Python code by running a SonarQube analysis.

To do so, follow these steps:

- Open the SonarQube dashboard.
- Create a new project and configure the project settings as needed: add the plugin and get your token.
- Run the analysis (`mvn org.sonarsource.scanner.maven:sonar-maven-plugin:3.9.1.2184:sonar -Dsonar.login=YOUR_TOKEN`).

_GreenPyData_ will then analyze your PyTorch code and flag any energy-intensive or computationally inefficient code
segments that can be optimized.

## Contribution

Any help or contribution to _GreenPyData_ is highly appreciated! Feel free to fork the repository, make your changes,
and submit a pull request.

If you encounter any issues or have suggestions for improvement, please open an issue.

For code style and formatting, please read https://github.com/SonarSource/sonar-developer-toolset.

## Implemented rules (PyTorch only for now)

The core idea for rules is "100% precision". Rules should not trigger false positives. The package should be used by 
Data Scientists to help them write _greener_ code and not bother them with thousands of false alarms.  

| ID  | Rule name                                         | Desc.                                                                              |
|-----|---------------------------------------------------|------------------------------------------------------------------------------------|
| P1  | AvoidDataParallelInsteadofDistributedDataParallel | Usage of DistributedDataParallel instead of DataParallel even for a single node    |  
| P2  | AvoidBlockingDataloaders                          | Usage of asynchronous data loading for better (and shorter) GPU usage              |
| P3  | AvoidNonPinnedMemoryForDataloaders                | Usage of pinned memory to reduce data transfer in RAM                              |
| P4  | AvoidConvBiasBeforeBatchNorm (Conv2d)             | Remove bias for convolutions before batch norm layers to save time and memory      |
| P5  | AvoidCreatingTensorUsingNumpyOrNativePython       | Directly create tensors as torch.Tensor and avoid Numpy or native python functions |
| P6  | UseInPlaceOperationsInModulesWhenPossible         | Use InPlace operations when possible (_only implemented for sequential modules_)   |

## Conclusion

Thank you for using _GreenPyData_! We hope it helps you in eco-designing your data science code and contributes to a
more sustainable software development process. If you have any questions or feedback, rules, ideas or anything else :) feel
free to reach out.
