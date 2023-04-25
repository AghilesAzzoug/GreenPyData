/*
 * SonarQube Python Plugin
 * Copyright (C) 2012-2019 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */
package fr.greenpydata.pytorch;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import fr.greenpydata.pytorch.checks.AvoidDataParallelInsteadofDistributedDataParallel;
import fr.greenpydata.pytorch.checks.AvoidBlockingDataloaders;
import fr.greenpydata.pytorch.checks.AvoidNonPinnedMemoryForDataloaders;
import fr.greenpydata.pytorch.checks.AvoidConvBiasBeforeBatchNorm;
import fr.greenpydata.pytorch.checks.AvoidCreatingTensorUsingNumpyOrNativePython;
import fr.greenpydata.pytorch.checks.UseInPlaceOperationsInModulesWhenPossible;

import org.sonar.api.server.rule.RulesDefinition;
import org.sonar.api.server.rule.RulesDefinitionAnnotationLoader;
import org.sonar.plugins.python.api.PythonCustomRuleRepository;

public class PytorchRuleRepository implements RulesDefinition, PythonCustomRuleRepository {

  public static final String LANGUAGE = "py";
  public static final String NAME = "Eco-Sonar Plugin for PyTorch Data Scientists";
  public static final String RESOURCE_BASE_PATH = "/fr/greenpydata/l10n/python/rules/python/";
  public static final String REPOSITORY_KEY = "gci-python";

  @Override
  public void define(Context context) {
    NewRepository repository = context.createRepository(repositoryKey(), LANGUAGE).setName(NAME);

    new RulesDefinitionAnnotationLoader().load(repository, checkClasses().toArray(new Class[] {}));

    // technical debt
    Map<String, String> remediationCosts = new HashMap<>();
    //remediationCosts.put(AvoidSQLRequestInLoop.RULE_KEY, "10min");
    //remediationCosts.put(AvoidDataParallelInsteadofDistributedDataParallel.RULE_KEY, "20min");
    repository.rules().forEach(rule -> {
      String debt = remediationCosts.get(rule.key());

      if (debt == null || debt.trim().equals("")) {
        // default debt to 5min for issue correction
        rule.setDebtRemediationFunction(
          rule.debtRemediationFunctions().constantPerIssue("5min"));
      } else {
        rule.setDebtRemediationFunction(
          rule.debtRemediationFunctions().constantPerIssue(debt));
      }
    });

    // HTML description
    repository.rules().forEach(rule ->
      rule.setHtmlDescription(loadResource(RESOURCE_BASE_PATH + rule.key() + ".html")));

    repository.done();
  }

  @Override
  public String repositoryKey() {
    return REPOSITORY_KEY;
  }

  @Override
  public List<Class> checkClasses() {
    return Arrays.asList(
      AvoidDataParallelInsteadofDistributedDataParallel.class,
      AvoidBlockingDataloaders.class,
      AvoidNonPinnedMemoryForDataloaders.class,
      AvoidConvBiasBeforeBatchNorm.class,
      AvoidCreatingTensorUsingNumpyOrNativePython.class,
      UseInPlaceOperationsInModulesWhenPossible.class
    );
  }

  private String loadResource(String path) {
    URL resource = getClass().getResource(path);
    if (resource == null) {
      throw new IllegalStateException("Resource not found: " + path);
    }
    ByteArrayOutputStream result = new ByteArrayOutputStream();
    try (InputStream in = resource.openStream()) {
      byte[] buffer = new byte[1024];
      for (int len = in.read(buffer); len != -1; len = in.read(buffer)) {
        result.write(buffer, 0, len);
      }
      return new String(result.toByteArray(), StandardCharsets.UTF_8);
    } catch (IOException e) {
      throw new IllegalStateException("Failed to read resource: " + path, e);
    }
  }
}
