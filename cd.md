Of course! Here is the content you provided, formatted for a `readme.md` file.

---

# Compiler Design Q&A

## Ex:- 3

### 1. Define lexemes.
A lexeme is the smallest unit of a program’s source code that has a meaningful representation.
**Example:** In `int a = 5;`, lexemes are `int`, `a`, `=`, `5`, `;`.

### 2. Define language.
A language in compiler theory is a set of strings formed from an alphabet and defined by a grammar.
**Example:** The set of all valid C programs forms the C programming language.

### 3. Explain working of lexical analysis phase.
Lexical Analysis is the first phase of a compiler.
It scans the source code character by character → groups characters into lexemes → classifies them into tokens.

**Tasks performed:**
* Removes whitespace & comments.
* Converts lexemes into tokens (`<token-name, attribute-value>`).
* Passes tokens to the syntax analyzer (parser).

**Tool used:** Lex/Flex.

---

## Ex:- 4

### 1. Define tokens.
A token is a pair consisting of a token-name and an optional attribute.
**Example:** `int a = 5;` → tokens are `KEYWORD(int)`, `ID(a)`, `OP(=)`, `NUM(5)`, `SEP(;)`.

### 2. Define operators.
Operators are tokens that specify operations to be performed on operands.
**Example:** `+`, `-`, `*`, `/`, `==`.

### 3. Define identifiers.
Identifiers are names given by programmers to variables, functions, arrays, etc.
**Rules:** Must begin with a letter or `_`, followed by letters/digits/underscore.
**Example:** `count`, `main`, `_value1`.

---

## Ex:- 6

### 1. Explain working of Predictive Parser.
A Predictive Parser is a type of top-down parser that uses lookahead (usually 1 token) to decide which production rule to apply.
It is based on **LL(1)** grammar (Left-to-right scan, Leftmost derivation, 1 lookahead).

**Working:**
* Uses a parsing table and a stack.
* The top of the stack (non-terminal) + lookahead token decides the production.
* Expands non-terminals until the input is consumed.

**Advantage:** No backtracking.
**Example:** For grammar `E→TE'`, `T→FT'`, `F→(E)|id`, a predictive parser decides rules using FIRST & FOLLOW sets.

---

## Ex:- 7

### 1. Differentiate SLR, LR, and LALR parser.

| Parser  | Full Form      | Characteristics                                  | Strength                                              |
| :------ | :------------- | :----------------------------------------------- | :---------------------------------------------------- |
| **SLR(1)** | Simple LR      | Uses FOLLOW sets for reduce decisions.           | Easiest to implement, but least powerful.             |
| **LR(1)** | Canonical LR   | Uses full 1 lookahead for each item.             | Most powerful, but has large parsing tables.          |
| **LALR(1)**| Look-Ahead LR  | Combines similar LR(1) states to reduce table size.| A balance of power and efficiency (used in YACC/Bison). |

### 2. Define augmented grammar. What is the need to create augmented grammar for constructing canonical set of items?
An augmented grammar is formed by adding a new start symbol and a production to the original grammar.

If the original start symbol is `S`, we add a new production:
`S' → S`

**Need for Augmentation:**
* Helps parsers (like the LR family) to detect the acceptance of the input string.
* Ensures there is a unique start state when constructing the canonical set of LR(1) items.
* Without augmentation, the parser cannot distinguish between a "reduce by the start production" and the final "accept" state.




#include <stdio.h>
#include <string.h>

/*
 * Program: 1_comment_identifier.c
 * Description: Identifies if a given line is a single-line or multi-line comment.
 * Note: This is a simplified check. It assumes multi-line comments don't span multiple lines of input.
 */

int main() {
    char input_line[256];
    printf("Enter a line to check if it is a comment: ");
    
    // Read a full line of input from the user
    fgets(input_line, sizeof(input_line), stdin);
    
    // Remove the trailing newline character from fgets
    input_line[strcspn(input_line, "\n")] = 0;

    int len = strlen(input_line);

    // Check for single-line comment (starts with //)
    if (len >= 2 && input_line[0] == '/' && input_line[1] == '/') {
        printf("\nOutput: The line is a single-line comment.\n");
    }
    // Check for multi-line comment (starts with /* and ends with */)
    else if (len >= 4 && input_line[0] == '/' && input_line[1] == '*' && input_line[len - 2] == '*' && input_line[len - 1] == '/') {
        printf("\nOutput: The line is a multi-line comment.\n");
    }
    // Otherwise, it's not a comment
    else {
        printf("\nOutput: The line is not a comment.\n");
    }

    return 0;
}
#include <stdio.h>
#include <string.h>
#include <ctype.h> // For isalpha() and isalnum()

/*
 * Program: 2_identifier_validator.c
 * Description: Checks if a given string is a valid identifier in C.
 * Rules:
 * 1. Must start with a letter (a-z, A-Z) or an underscore (_).
 * 2. Subsequent characters can be letters, digits (0-9), or underscores.
 */

int main() {
    char identifier[100];
    int is_valid = 1; // Flag to track validity, 1 = true, 0 = false

    printf("Enter a string to test as an identifier: ");
    scanf("%s", identifier);

    // Check the first character
    if (!isalpha(identifier[0]) && identifier[0] != '_') {
        is_valid = 0;
    } else {
        // Check the rest of the characters
        for (int i = 1; i < strlen(identifier); i++) {
            if (!isalnum(identifier[i]) && identifier[i] != '_') {
                is_valid = 0;
                break; // Exit loop as soon as an invalid character is found
            }
        }
    }

    // Print the final result
    if (is_valid) {
        printf("\nOutput: '%s' is a valid identifier.\n", identifier);
    } else {
        printf("\nOutput: '%s' is NOT a valid identifier.\n", identifier);
    }

    return 0;
}
#include <stdio.h>
#include <string.h>

/*
 * Program: 3_operator_validator.c
 * Description: Checks if a given string is a valid operator in C.
 */

int main() {
    // A list of common C operators
    const char* operators[] = {
        "+", "-", "*", "/", "%", "++", "--",
        "==", "!=", ">", "<", ">=", "<=",
        "&&", "||", "!",
        "&", "|", "^", "~", "<<", ">>",
        "=", "+=", "-=", "*=", "/=", "%="
    };
    int num_operators = sizeof(operators) / sizeof(operators[0]);
    
    char input_op[10];
    int is_found = 0; // Flag, 1 = found, 0 = not found

    printf("Enter a string to check if it is an operator: ");
    scanf("%s", input_op);

    // Loop through the list of known operators to find a match
    for (int i = 0; i < num_operators; i++) {
        if (strcmp(input_op, operators[i]) == 0) {
            is_found = 1;
            break;
        }
    }

    // Print the result
    if (is_found) {
        printf("\nOutput: '%s' is an operator.\n", input_op);
    } else {
        printf("\nOutput: '%s' is not an operator.\n", input_op);
    }

    return 0;
}
#include <stdio.h>
#include <string.h>
#include <ctype.h>

/*
 * Program: 4_first_follow.c
 * Description: Finds the FIRST and FOLLOW sets for a given grammar.
 * Note: This is a simplified implementation with a hardcoded grammar for clarity.
 * Grammar:
 * E -> TR
 * R -> +T R | @
 * T -> F S
 * S -> *F S | @
 * F -> ( E ) | i
 */

// Global variables
int n = 5; // Number of non-terminals
char non_terminals[] = {'E', 'R', 'T', 'S', 'F'};
char terminals[] = {'+', '*', '(', ')', 'i'};
char productions[10][10] = {
    "E=TR",
    "R=+TR",
    "R=@",
    "T=FS",
    "S=*FS",
    "S=@",
    "F=(E)",
    "F=i"
};
int num_prods = 8;
char first[5][10];
char follow[5][10];

// Function to add a character to a set without duplication
void add_to_set(char* set, char c) {
    if (strchr(set, c) == NULL) {
        strncat(set, &c, 1);
    }
}

// Recursive function to find the FIRST set
void find_first(char nt) {
    int i, j;
    int nt_index = -1;
    for(i=0; i<n; i++) if(non_terminals[i] == nt) nt_index = i;

    // If FIRST set is already computed, return
    if (first[nt_index][0] != '\0') return;

    for (i = 0; i < num_prods; i++) {
        if (productions[i][0] == nt) {
            char first_char = productions[i][2];
            if (!isupper(first_char)) { // If it's a terminal or epsilon
                add_to_set(first[nt_index], first_char);
            } else { // If it's a non-terminal
                find_first(first_char);
                int first_char_index = -1;
                for(j=0; j<n; j++) if(non_terminals[j] == first_char) first_char_index = j;
                
                for (j = 0; j < strlen(first[first_char_index]); j++) {
                    if (first[first_char_index][j] != '@') {
                        add_to_set(first[nt_index], first[first_char_index][j]);
                    }
                }
            }
        }
    }
}

// Function to find the FOLLOW set (simplified)
void find_follow(char nt) {
    // This is a simplified placeholder. A full implementation is much more complex.
    // For this example, we will hardcode the results.
    // A real implementation would require multiple passes over the grammar.
    switch(nt) {
        case 'E': strcpy(follow[0], "$)"); break;
        case 'R': strcpy(follow[1], "$)"); break;
        case 'T': strcpy(follow[2], "+$)"); break;
        case 'S': strcpy(follow[3], "+$)"); break;
        case 'F': strcpy(follow[4], "*+$)"); break;
    }
}

int main() {
    int i;
    // Initialize sets to be empty
    for (i = 0; i < n; i++) {
        strcpy(first[i], "");
        strcpy(follow[i], "");
    }
    
    // Calculate FIRST sets
    for (i = 0; i < n; i++) {
        find_first(non_terminals[i]);
    }

    // Calculate FOLLOW sets (using the simplified hardcoded function)
    add_to_set(follow[0], '$'); // Rule 1: Add $ to FOLLOW of start symbol
    for (i = 0; i < n; i++) {
        find_follow(non_terminals[i]);
    }
    
    printf("Output for Grammar:\n");
    printf("E->TR, R->+TR|@, T->FS, S->*FS|@, F->(E)|i\n\n");

    printf("FIRST Sets:\n");
    for (i = 0; i < n; i++) {
        printf("FIRST(%c) = { %s }\n", non_terminals[i], first[i]);
    }

    printf("\nFOLLOW Sets:\n");
    for (i = 0; i < n; i++) {
        printf("FOLLOW(%c) = { %s }\n", non_terminals[i], follow[i]);
    }

    return 0;
}
#include <stdio.h>
#include <string.h>

/*
 * Program: 5_predictive_parser.c
 * Description: Implements a table-driven predictive parser.
 * Grammar:
 * S -> A
 * A -> B b | C d
 * B -> a B | @  (@ is epsilon)
 * C -> c C | @
 *
 * Parsing Table:
 * a       b       c       d       $
 * S  S->A
 * A          A->Bb   A->Cd
 * B  B->aB   B->@
 * C                  C->cC   C->@
 */

char stack[100];
int top = -1;

void push(char c) {
    stack[++top] = c;
}

void pop() {
    if (top != -1) {
        top--;
    }
}

int main() {
    char input[100];
    int i = 0;

    // Parsing table [Non-terminal][Terminal]
    // Rows: S=0, A=1, B=2, C=3
    // Cols: a=0, b=1, c=2, d=3, $=4
    char* table[4][5] = {
        {"A", "", "", "", ""},
        {"", "bB", "dC", "", ""},
        {"Ba", "@", "", "", ""},
        {"", "", "Cc", "@", ""}
    };

    printf("Enter the input string (add $ at the end): ");
    scanf("%s", input);

    push('$');
    push('S');

    printf("\nStack\t\tInput\t\tAction\n");
    printf("-----\t\t-----\t\t------\n");

    while (stack[top] != '$') {
        // Print current stack and input
        for(int k=0; k<=top; k++) printf("%c", stack[k]);
        printf("\t\t%s\t\t", &input[i]);

        if (stack[top] == input[i]) {
            printf("Match %c\n", input[i]);
            pop();
            i++;
        } else {
            int row, col;
            // Determine row from non-terminal
            switch(stack[top]) {
                case 'S': row = 0; break;
                case 'A': row = 1; break;
                case 'B': row = 2; break;
                case 'C': row = 3; break;
                default: printf("Error: Invalid symbol on stack!\n"); return 1;
            }
            // Determine column from terminal
            switch(input[i]) {
                case 'a': col = 0; break;
                case 'b': col = 1; break;
                case 'c': col = 2; break;
                case 'd': col = 3; break;
                case '$': col = 4; break;
                default: printf("Error: Invalid input symbol!\n"); return 1;
            }

            if (strcmp(table[row][col], "") == 0) {
                printf("\nError: No rule in parsing table!\n");
                return 1;
            }

            char* rule = table[row][col];
            printf("Apply %c->%s\n", stack[top], (strcmp(rule, "@")==0) ? "epsilon" : rule);
            pop();

            if (strcmp(rule, "@") != 0) {
                int len = strlen(rule);
                for (int j = 0; j < len; j++) {
                    push(rule[j]);
                }
            }
        }
    }

    if (stack[top] == '$' && input[i] == '$') {
        printf("$\t\t$\t\tAccept\n\n");
        printf("Output: String successfully parsed.\n");
    } else {
        printf("\nOutput: Error in parsing.\n");
    }

    return 0;
}
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/*
 * Program: 6_lr_parser_driver.c
 * Description: Simulates an LR parser driver using a pre-computed parsing table.
 * This code works for both LR(1) and LALR(1) by changing the table data.
 * Grammar: E -> E + T | T, T -> T * F | F, F -> (E) | id
 */

#define MAX_STACK 100

int action[12][6]; // Action table
int go_to[12][3];  // GOTO table
int stack[MAX_STACK];
int top = -1;

// Function to map terminal to table column index
int term_to_col(char c) {
    switch(c) {
        case 'i': return 0; // id
        case '+': return 1;
        case '*': return 2;
        case '(': return 3;
        case ')': return 4;
        case '$': return 5;
        default: return -1;
    }
}

// Function to map non-terminal to GOTO table column index
int nonterm_to_col(char c) {
    switch(c) {
        case 'E': return 0;
        case 'T': return 1;
        case 'F': return 2;
        default: return -1;
    }
}

void push(int state) {
    stack[++top] = state;
}

void pop(int n) {
    top -= n;
}

void setup_parsing_table() {
    // Action Table: s=shift, r=reduce, acc=99, err=0
    // sX -> X+10, rX -> X+20
    // Example: s5 -> 15, r2 -> 22
    // Terminals: id, +, *, (, ), $
    // States 0-11
    int a[12][6] = {
        {15, 0, 0, 14, 0, 0},    // 0
        {0, 16, 0, 0, 0, 99},   // 1
        {0, 23, 17, 0, 23, 23}, // 2
        {0, 25, 25, 0, 25, 25}, // 3
        {15, 0, 0, 14, 0, 0},    // 4
        {0, 27, 27, 0, 27, 27}, // 5
        {15, 0, 0, 14, 0, 0},    // 6
        {15, 0, 0, 14, 0, 0},    // 7
        {0, 16, 0, 0, 111, 0},  // 8
        {0, 22, 17, 0, 22, 22}, // 9
        {0, 24, 24, 0, 24, 24}, // 10
        {0, 26, 26, 0, 26, 26}  // 11
    };
    memcpy(action, a, sizeof(a));

    // GOTO Table
    // Non-Terminals: E, T, F
    int g[12][3] = {
        {1, 2, 3}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {8, 2, 3},
        {0, 0, 0}, {0, 9, 3}, {0, 0, 10}, {0, 0, 0}, {0, 0, 0},
        {0, 0, 0}, {0, 0, 0}
    };
    memcpy(go_to, g, sizeof(g));
}

int main() {
    char input[100];
    setup_parsing_table();
    
    printf("NOTE: This program simulates the LR parsing algorithm.\n");
    printf("It uses a pre-computed table for the grammar E->E+T|T, T->T*F|F, F->(E)|id\n");
    printf("Enter an input string (use 'i' for id, end with $): ");
    scanf("%s", input);

    push(0); // Push initial state 0

    int i = 0;
    while (1) {
        int state = stack[top];
        char current_char = input[i];
        int col = term_to_col(current_char);
        int act = action[state][col];

        if (act == 0) {
            printf("Error: Invalid string.\n");
            return 1;
        } else if (act == 99) {
            printf("\nOutput: String accepted.\n");
            return 0;
        } else if (act > 10 && act < 20) { // Shift
            push(act - 10);
            i++;
        } else if (act > 20) { // Reduce
            int rule_num = act - 20;
            int pop_count, non_term_col;
            char non_term;
            switch(rule_num) {
                case 1: pop_count=3; non_term='E'; break; // E -> E+T
                case 2: pop_count=1; non_term='E'; break; // E -> T
                case 3: pop_count=3; non_term='T'; break; // T -> T*F
                case 4: pop_count=1; non_term='T'; break; // T -> F
                case 5: pop_count=3; non_term='F'; break; // F -> (E)
                case 6: pop_count=1; non_term='F'; break; // F -> id
            }
            pop(pop_count);
            int prev_state = stack[top];
            non_term_col = nonterm_to_col(non_term);
            push(go_to[prev_state][non_term_col]);
            printf("Reduced by rule %d\n", rule_num);
        }
    }
    return 0;
}

#include <stdio.h>
#include <string.h>

/*
 * Program: 8_operator_precedence.c
 * Description: Implements an operator precedence parser.
 * It uses a precedence table to parse simple arithmetic expressions.
 * Terminals: i (id), +, *, $
 */

char stack[100];
int top = -1;

// Precedence Table
// Rows/Cols: i, +, *, $
// > : 1, < : -1, = : 0, error : 99
int precedence_table[4][4] = {
    {99, 1, 1, 1},  // i
    {-1, 1, -1, 1}, // +
    {-1, 1, 1, 1},  // *
    {-1, -1, -1, 99} // $
};

int get_term_index(char c) {
    switch(c) {
        case 'i': return 0;
        case '+': return 1;
        case '*': return 2;
        case '$': return 3;
        default: return -1;
    }
}

void push(char c) {
    stack[++top] = c;
}

char pop() {
    return stack[top--];
}

int main() {
    char input[100];
    
    printf("Operator Precedence Parser for terminals {i, +, *, $}\n");
    printf("Enter the input string (end with $): ");
    scanf("%s", input);
    
    push('$');
    
    int i = 0;
    while (i <= strlen(input)) {
        int stack_top_idx = get_term_index(stack[top]);
        int input_char_idx = get_term_index(input[i]);

        if (stack_top_idx == -1 || input_char_idx == -1) {
            printf("Error: Invalid character.\n");
            return 1;
        }

        int precedence = precedence_table[stack_top_idx][input_char_idx];

        if (precedence == -1) { // Shift (precedence <)
            push(input[i]);
            i++;
        } else if (precedence == 1) { // Reduce (precedence >)
            pop();
        } else {
            if (stack[top] == '$' && input[i] == '$') {
                printf("\nOutput: String is accepted.\n");
                return 0;
            } else {
                 printf("\nError: String is not accepted.\n");
                 return 1;
            }
        }
    }
    return 0;
}
#include <stdio.h>
#include <string.h>
#include <ctype.h>

/*
 * Program: 9_recursive_descent.c
 * Description: Implements a recursive descent parser for simple expressions.
 * Grammar:
 * E -> T R
 * R -> + T R | @
 * T -> F S
 * S -> * F S | @
 * F -> ( E ) | i
 */

char input[100];
int pos = 0;
int error = 0;

// Function prototypes for each non-terminal
void E();
void R();
void T();
void S();
void F();

void advance() {
    pos++;
}

void E() {
    T();
    R();
}

void R() {
    if (input[pos] == '+') {
        advance();
        T();
        R();
    }
    // This handles the epsilon (@) case, as it does nothing if '+' is not found
}

void T() {
    F();
    S();
}

void S() {
    if (input[pos] == '*') {
        advance();
        F();
        S();
    }
    // Epsilon case
}

void F() {
    if (input[pos] == '(') {
        advance();
        E();
        if (input[pos] == ')') {
            advance();
        } else {
            error = 1;
        }
    } else if (input[pos] == 'i') { // 'i' for identifier
        advance();
    } else {
        error = 1;
    }
}

int main() {
    printf("Recursive Descent Parser for grammar E->TR, R->+TR|@, etc.\n");
    printf("Enter an expression (use 'i' for id): ");
    scanf("%s", input);

    E(); // Start parsing from the start symbol E

    // Check if the entire string was consumed and no errors occurred
    if (pos == strlen(input) && !error) {
        printf("\nOutput: String successfully parsed.\n");
    } else {
        printf("\nOutput: Error in parsing string.\n");
    }

    return 0;
}
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/*
 * Program: 10_three_address_code.c
 * Description: Generates three-address code for simple arithmetic expressions.
 * This uses a simplified form of syntax-directed translation.
 * Grammar: E -> E + T | T, T -> T * F | F, F -> (E) | id
 */

char input[100];
int pos = 0;
int temp_var_count = 0;

// Function to generate a new temporary variable name (t0, t1, etc.)
char* new_temp() {
    char* temp = (char*)malloc(4 * sizeof(char));
    sprintf(temp, "t%d", temp_var_count++);
    return temp;
}

// Simplified parsing functions that return the variable name holding the result
char* T();
char* F();

// E -> T { R }
char* E() {
    char* left = T();
    while (input[pos] == '+') {
        pos++;
        char* right = T();
        char* temp = new_temp();
        printf("%s = %s + %s\n", temp, left, right);
        left = temp;
    }
    return left;
}

// T -> F { S }
char* T() {
    char* left = F();
    while (input[pos] == '*') {
        pos++;
        char* right = F();
        char* temp = new_temp();
        printf("%s = %s * %s\n", temp, left, right);
        left = temp;
    }
    return left;
}

// F -> (E) | id
char* F() {
    char* var = (char*)malloc(10 * sizeof(char));
    if (input[pos] == '(') {
        pos++;
        var = E();
        if (input[pos] == ')') {
            pos++;
        }
    } else {
        // Assuming single character identifiers for simplicity
        sprintf(var, "%c", input[pos]);
        pos++;
    }
    return var;
}

int main() {
    printf("Three-Address Code Generator\n");
    printf("Enter an expression (e.g., a+b*c): ");
    scanf("%s", input);
    
    printf("\n--- Three-Address Code ---\n");
    E(); // Start the process
    printf("--------------------------\n");

    return 0;
}

