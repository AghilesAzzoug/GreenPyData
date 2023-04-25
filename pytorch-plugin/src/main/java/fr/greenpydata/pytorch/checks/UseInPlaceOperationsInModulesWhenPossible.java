package fr.greenpydata.pytorch.checks;

import org.sonar.check.Priority;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Argument;

import static org.sonar.plugins.python.api.tree.Tree.Kind.NAME;
import static org.sonar.plugins.python.api.tree.Tree.Kind.REGULAR_ARGUMENT;
import static org.sonar.plugins.python.api.tree.Tree.Kind.CALL_EXPR;

@Rule(
  key = UseInPlaceOperationsInModulesWhenPossible.RULE_KEY,
  name = UseInPlaceOperationsInModulesWhenPossible.MESSAGE,
  description = UseInPlaceOperationsInModulesWhenPossible.MESSAGE,
  priority = Priority.MINOR,
  tags = {"bug", "eco-design"})
public class UseInPlaceOperationsInModulesWhenPossible extends PythonSubscriptionCheck {

  public static final String RULE_KEY = "P6";
  private static final String sequentialQualifiedName = "torch.nn.Sequential";
  protected static final String MESSAGE = "Use InPlace operations when possible.";

  private void reportIfNotInPlace(SubscriptionContext context, CallExpression module) {
    String moduleName = Utils.getQualifiedName(module);
    int argPosition;
    switch (moduleName) {
      case "torch.nn.ReLU":
      case "torch.nn.Hardsigmoid":
      case "torch.nn.Hardwish":
      case "torch.nn.Mish":
      case "torch.nn.ReLU6":
      case "torch.nn.SELU":
      case "torch.nn.SiLU":
        argPosition = 0;
        break;
      case "torch.nn.AlphaDropout":
      case "torch.nn.CELU":
      case "torch.nn.Dropout":
      case "torch.nn.Dropout1d":
      case "torch.nn.Dropout2d":
      case "torch.nn.Dropout3d":
      case "torch.nn.ELU":
      case "torch.nn.FeatureAlphaDropout":
      case "torch.nn.LeakyReLU":
        argPosition = 1;
        break;
      case "torch.nn.Hardtanh":
      case "torch.nn.RReLU":
      case "torch.nn.Threshold":
        argPosition = 2;
        break;
      default:
        return;
    }

    RegularArgument argument = Utils.nthArgumentOrKeyword(argPosition, "inplace", module.arguments());

    if (argument == null) {
      // no argument means not an inplace operation
      context.addIssue(module, MESSAGE);
    } else if (argument.expression().is(NAME) && ((Name) argument.expression()).name().equals("False")) {
      context.addIssue(module, MESSAGE);
    }
  }

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ctx -> {
      CallExpression callExpression = (CallExpression) ctx.syntaxNode();
      if (Utils.getQualifiedName(callExpression).equals(sequentialQualifiedName)) {
        int moduleIndex = 1; // skip the first
        int nModulesInSequential = Utils.getArgumentsFromCall(callExpression).size();
        while (moduleIndex < nModulesInSequential) {
          Argument moduleInSequential = Utils.getArgumentsFromCall(callExpression).get(moduleIndex);
          if (moduleInSequential.is(REGULAR_ARGUMENT) && ((RegularArgument) moduleInSequential).expression().is(CALL_EXPR)) {
            CallExpression module = (CallExpression) ((RegularArgument) moduleInSequential).expression();
            reportIfNotInPlace(ctx, module);
          }
          moduleIndex++;
        }
      }
    });
  }
}
