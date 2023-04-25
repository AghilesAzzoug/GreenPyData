package fr.greenpydata.pytorch.checks;

import org.sonar.check.Priority;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.NumericLiteral;

import java.util.Optional;

import static org.sonar.plugins.python.api.tree.Tree.Kind.NUMERIC_LITERAL;

@Rule(
  key = AvoidBlockingDataloaders.RULE_KEY,
  name = AvoidBlockingDataloaders.MESSAGE,
  description = AvoidBlockingDataloaders.MESSAGE,
  priority = Priority.MINOR,
  tags = {"bug", "eco-design"})
public class AvoidBlockingDataloaders extends PythonSubscriptionCheck {

  public static final String RULE_KEY = "P2";
  private static final String dataloaderFullyQualifiedName = "torch.utils.data.DataLoader";
  private static final int numWorkersArgumentPosition = 5;
  private static final String numWorkersArgumentName = "num_workers";
  protected static final String MESSAGE = "Use asynchronous data loading for better GPU usage.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ctx -> {
      CallExpression callExpression = (CallExpression) ctx.syntaxNode();

      if (getQualifiedName(callExpression).equals(dataloaderFullyQualifiedName)) {
        RegularArgument numWorkersArgument = Utils.nthArgumentOrKeyword(numWorkersArgumentPosition,
          numWorkersArgumentName,
          callExpression.arguments());

        if (numWorkersArgument == null) {
          ctx.addIssue(callExpression, MESSAGE);
        } else {
          Optional.of(numWorkersArgument).filter(this::checkBadValuesForNumWorkers)
            .ifPresent(arg -> ctx.addIssue(arg, MESSAGE));
        }
      }
    });
  }

  private String getQualifiedName(CallExpression callExpression) {
    Symbol symbol = callExpression.calleeSymbol();
    return symbol != null && symbol.fullyQualifiedName() != null ? symbol.fullyQualifiedName() : "";
  }

  private boolean checkBadValuesForNumWorkers(RegularArgument numWorkersArgument) {
    Expression expression = numWorkersArgument.expression();

    try {
      return expression.is(NUMERIC_LITERAL) && ((NumericLiteral) expression).valueAsLong() <= 0;
    } catch (NumberFormatException nfe) {
      return false;
    }
  }
}
