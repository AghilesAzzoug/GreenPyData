package fr.greenpydata.pytorch.checks;

import org.junit.Test;
import org.sonar.python.checks.utils.PythonCheckVerifier;

public class AvoidConvBiasBeforeBatchNormTest {

  @Test
  public void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/AvoidConvBiasBeforeBatchNorm.py", new AvoidConvBiasBeforeBatchNorm());
  }
}
