package fr.greenpydata.pytorch.checks;

import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.CallExpression;

import javax.annotation.CheckForNull;
import java.util.List;
import java.util.Objects;

public class Utils {

  private static boolean hasKeyword(Argument argument, String keyword) {
    if (!argument.is(new Tree.Kind[] {Tree.Kind.REGULAR_ARGUMENT})) {
      return false;
    } else {
      Name keywordArgument = ((RegularArgument) argument).keywordArgument();
      return keywordArgument != null && keywordArgument.name().equals(keyword);
    }
  }

  @CheckForNull
  public static RegularArgument nthArgumentOrKeyword(int argPosition, String keyword, List<Argument> arguments) {
    for (int i = 0; i < arguments.size(); ++i) {
      Argument argument = (Argument) arguments.get(i);
      if (hasKeyword(argument, keyword)) {
        return (RegularArgument) argument;
      }

      if (argument.is(new Tree.Kind[] {Tree.Kind.REGULAR_ARGUMENT})) {
        RegularArgument regularArgument = (RegularArgument) argument;
        if (regularArgument.keywordArgument() == null && argPosition == i) {
          return regularArgument;
        }
      }
    }

    return null;
  }

  public static String getQualifiedName(CallExpression callExpression) {
    Symbol symbol = callExpression.calleeSymbol();
    return symbol != null && symbol.fullyQualifiedName() != null ? symbol.fullyQualifiedName() : "";
  }

  public static String getMethodName(CallExpression callExpression) {
    Symbol symbol = callExpression.calleeSymbol();
    return symbol != null && symbol.name() != null ? symbol.name() : "";
  }

  public static List<Argument> getArgumentsFromCall(CallExpression callExpression) {
    try {
      return Objects.requireNonNull(callExpression.argumentList()).arguments();
    } catch (NullPointerException e) {
      return List.of();
    }
  }
}
