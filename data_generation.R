
# ÉTAPE 1: DATA GENERATION - MARKOV CHAINS


# Set seed for reproducibility
set.seed(42)

# ÉTAPE 1: Define states


states <- c("sain_non_vaccine", "retabli", "contamine", "mort", "sain_vaccine", "infecte")
n_states <- length(states)
cat("States available:", paste(states, collapse=", "), "\n")


# ÉTAPE 2: Create transition matrix M1 (Uniform)


M1 <- matrix(runif(n_states * n_states), nrow = n_states, ncol = n_states)
# Normalize: each row sums to 1
M1 <- M1 / rowSums(M1)

cat("\nTransition Matrix M1 (Uniform):\n")
print(M1)


# ÉTAPE 3: Generate sequences with M1


generate_sequences <- function(transition_matrix, n_sequences = 150, seq_length = 10) {
  """
  Generate sequences using Markov chain transition matrix
  
  Args:
    transition_matrix: n_states x n_states matrix
    n_sequences: number of sequences to generate
    seq_length: length of each sequence
  
  Returns:
    sequences: matrix (n_sequences, seq_length) with state indices
  """
  n_states <- nrow(transition_matrix)
  sequences <- matrix(0, nrow = n_sequences, ncol = seq_length)
  
  for (i in 1:n_sequences) {
    # Random initial state
    current_state <- sample(1:n_states, 1)
    sequences[i, 1] <- current_state
    
    # Generate rest of sequence
    for (t in 2:seq_length) {
      # Transition probabilities from current state
      probs <- transition_matrix[current_state, ]
      # Choose next state according to probabilities
      next_state <- sample(1:n_states, 1, prob = probs)
      sequences[i, t] <- next_state
      current_state <- next_state
    }
  }
  
  return(sequences)
}

# Generate sequences
sequences_m1 <- generate_sequences(M1, n_sequences = 150, seq_length = 10)

cat("\nSequences shape:", nrow(sequences_m1), "x", ncol(sequences_m1), "\n")
cat("First 5 sequences (indices):\n")
print(sequences_m1[1:5, ])


# ÉTAPE 4: Convert to state names


sequences_m1_names <- matrix(0, nrow = nrow(sequences_m1), ncol = ncol(sequences_m1), 
                             dimnames = list(NULL, paste0("Day_", 0:9)))

for (i in 1:nrow(sequences_m1)) {
  for (j in 1:ncol(sequences_m1)) {
    sequences_m1_names[i, j] <- states[sequences_m1[i, j]]
  }
}

cat("\nFirst 3 sequences (state names):\n")
for (i in 1:3) {
  cat(paste0("Seq ", i-1, ": ", paste(sequences_m1_names[i, ], collapse = " → "), "\n"))
}


# ÉTAPE 5: Save data


# Save as CSV
write.csv(sequences_m1, "sequences_m1.csv", row.names = FALSE)

cat("\n✓ Data saved to 'sequences_m1.csv'\n")
