package fr.greenpydata.pytorch.checks;

import org.sonar.check.Priority;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;

import java.util.Optional;

import static org.sonar.plugins.python.api.tree.Tree.Kind.NAME;

@Rule(
  key = AvoidNonPinnedMemoryForDataloaders.RULE_KEY,
  name = AvoidNonPinnedMemoryForDataloaders.MESSAGE,
  description = AvoidNonPinnedMemoryForDataloaders.MESSAGE,
  priority = Priority.MINOR,
  tags = {"bug", "eco-design"})
public class AvoidNonPinnedMemoryForDataloaders extends PythonSubscriptionCheck {

  public static final String RULE_KEY = "P3";
  private static final String dataloaderFullyQualifiedName = "torch.utils.data.DataLoader";
  private static final int pinMemoryArgumentPosition = 7;
  private static final String pinMemoryArgumentName = "pin_memory";
  protected static final String MESSAGE = "Use pinned memory to reduce data transfer in RAM.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ctx -> {
      CallExpression callExpression = (CallExpression) ctx.syntaxNode();

      if (Utils.getQualifiedName(callExpression).equals(dataloaderFullyQualifiedName)) {
        RegularArgument numWorkersArgument = Utils.nthArgumentOrKeyword(pinMemoryArgumentPosition,
          pinMemoryArgumentName,
          callExpression.arguments());

        if (numWorkersArgument == null) {
          ctx.addIssue(callExpression, MESSAGE);
        } else {
          Optional.of(numWorkersArgument).filter(this::checkBadValuesForPinMemory)
            .ifPresent(arg -> ctx.addIssue(arg, MESSAGE));
        }
      }
    });
  }

  private boolean checkBadValuesForPinMemory(RegularArgument pinMemoryArgument) {
    Expression expression = pinMemoryArgument.expression();
    return expression.is(NAME) && ((Name) expression).name().equals("False");
  }
}
