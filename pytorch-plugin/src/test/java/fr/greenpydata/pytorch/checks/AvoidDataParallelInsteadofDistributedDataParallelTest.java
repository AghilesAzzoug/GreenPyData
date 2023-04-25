package fr.greenpydata.pytorch.checks;

import org.junit.Test;
import org.sonar.python.checks.utils.PythonCheckVerifier;

public class AvoidDataParallelInsteadofDistributedDataParallelTest {

  @Test
  public void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/AvoidDataParallelInsteadofDistributedDataParallel.py", new AvoidDataParallelInsteadofDistributedDataParallel());
  }
}
