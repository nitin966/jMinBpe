package tokenpair;

import java.util.Objects;

/**
 * Represents a pair of token IDs.
 */
public class TokenPair {
    private final int first;
    private final int second;

    /**
     * Constructs a new TokenPair.
     *
     * @param first The first token ID.
     * @param second The second token ID.
     */
    public TokenPair(int first, int second) {
        this.first = first;
        this.second = second;
    }

    public int getFirst() { return first; }
    public int getSecond() { return second; }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        TokenPair tokenPair = (TokenPair) o;
        return first == tokenPair.first && second == tokenPair.second;
    }

    @Override
    public int hashCode() {
        return Objects.hash(first, second);
    }

    @Override
    public String toString() {
        return "(" + first + ", " + second + ")";
    }
}
