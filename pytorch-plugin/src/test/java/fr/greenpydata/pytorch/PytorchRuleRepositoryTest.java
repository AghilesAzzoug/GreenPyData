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

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.Test;
import org.sonar.api.server.rule.RulesDefinition;

public class PytorchRuleRepositoryTest {

  @Test
  public void test() {
    PytorchRuleRepository pytorchRuleRepository = new PytorchRuleRepository();
    RulesDefinition.Context context = new RulesDefinition.Context();
    pytorchRuleRepository.define(context);
    assertThat(pytorchRuleRepository.repositoryKey()).isEqualTo(PytorchRuleRepository.REPOSITORY_KEY);
    assertThat(context.repositories()).hasSize(1).extracting("key").containsExactly(pytorchRuleRepository.repositoryKey());
    assertThat(context.repositories().get(0).rules()).hasSize(6);
    assertThat(pytorchRuleRepository.checkClasses()).hasSize(6);
  }
}
