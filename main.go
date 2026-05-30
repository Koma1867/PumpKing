// Name: 				PumpKing
// Author: 				Otto Laukkanen
//
// Board:   			0x88 with precomputed leaper tables
// Search:  			Alpha-Beta + Iterative Deepening + Quiescence + Check Extensions + TT
// Prune:   			Late Move Reductions, Null Move Pruning
// Eval:    			Tapered HCE (Material + PSTs, Tempo, Rook-file bonus, Doubled pawn, Bishop pair, isolated pawn)

package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync/atomic"
	"time"
	"unsafe"
)

// -------------------------
// --- Constants & Types ---
// -------------------------

// Piece constants
const (
	Empty   = 0
	WPawn   = 1
	WKnight = 2
	WBishop = 3
	WRook   = 4
	WQueen  = 5
	WKing   = 6
	BPawn   = 7
	BKnight = 8
	BBishop = 9
	BRook   = 10
	BQueen  = 11
	BKing   = 12
)

// Color constants
const (
	White = 0
	Black = 1
)

// Search constants
const (
	infScore             = 32000                      // Score larger than any eval
	mateScore            = 30000                      // Base mate score
	timeCheckMask        = 511                        // Clock check node count
	maxSearchDepth       = 64                         // Max search depth
	maxMoves             = 256                        // Max legal moves
	aspirationDelta      = 25                         // Aspiration window initial half-size
	aspirationMaxRetries = 8                          // Max aspiration retries
	lmrMinLegalMoves     = 3                          // min LMR moves
	lmrMinDepth          = 3                          // min LMR depth
	lmrDepthBumpAt       = 6                          // Reduction + 1
	lmrLegalBumpAt       = 6                          // Reduction + 1
	NullMoveReduction    = 2                          // Reduction amount for NMP
	MovesLeft            = 40                         // When movestogo = 0
	movetimeSafetyMs     = 50                         // Taken from time allocated to account for overhead
	mateThreshold        = mateScore - maxSearchDepth // Score for mate detection
)

// Search state
var (
	nodes       uint64      // Positions searched
	stopTime    time.Time   // When search should stop
	useTime     bool        // Whether time management is active
	abortSearch atomic.Bool // Set to true to terminate search

	searchDone chan struct{} // Closed after search has completed
)

// Waits for current search to finish
func waitForSearch() {
	if searchDone != nil { // Check if search is active
		abortSearch.Store(true) // Make search stop
		<-searchDone            // Waits for search to exit
		searchDone = nil        // Set back to nil
	}
}

// History heuristic
// (side)
// (from)
// (to)
var history [2][128][128]int

// Killer moves
// (ply)
// (0=primary, 1=secondary)
var killers [maxSearchDepth][2]Move

// Castle rights bits
const (
	WKCastle = 1
	WQCastle = 2
	BKCastle = 4
	BQCastle = 8
)

// Move flags
const (
	FlagNone       = 0
	FlagDoublePush = 1
	FlagCastle     = 2
	FlagEP         = 3
	FlagPromotion  = 4
)

// Bound flags
const (
	ttExact uint8 = iota // Exact score (PV node)
	ttAlpha              // Upper bound (failed low)
	ttBeta               // Lower bound (beta cutoff)
)

// TT sizes
const (
	ttMinMB = 1
	ttMaxMB = 1024
	ttDefMB = 16
)

// What TT entry stores
type TTEntry struct {
	Hash  uint64 // Full hash
	Move  Move   // Best move found at dis depth
	Score int16  // Stored score
	Depth int8   // Depth this entry was searched at
	Flag  uint8  // ttExact / ttAlpha / ttBeta
}

// TT mask & slice
var tt []TTEntry
var ttMask uint64

// Move encoded as 32 bits
type Move uint32

// Invalid or null move
const NoMove Move = 0

// Material & phase values
// (Empty) (WP) (WN) (WB) (WR) (WQ) (WK) (BP) (BN) (BB) (BR) (BQ) (BK)
var materialMg = [13]int{0, 95, 310, 340, 490, 920, 0, 95, 310, 340, 490, 920, 0}
var materialEg = [13]int{0, 115, 280, 305, 540, 945, 0, 115, 280, 305, 540, 945, 0}
var phaseValues = [13]int{0, 0, 1, 1, 2, 4, 0, 0, 1, 1, 2, 4, 0}

// Piece square tables
var pstMg [13][128]int
var pstEg [13][128]int

// Ordering constants
const (
	scorePVMove   = 10_000_000        // PV move
	scoreCapBase  = 1_000_000         // Base score for captures
	scoreKillerA  = scoreCapBase + 50 // Killer A slot, just below worst capture
	scoreKillerB  = scoreCapBase + 10 // Killer B slot
	mvvMultiplier = 100               // Victim value multiplier
)

// MVV-LVA lookup values
var mvvLvaVal = [13]int{0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0}

// Evaluation constants
const (
	maxPhase           = 24  // Max game phase
	tempoBonus         = 15  // Bonus for side to move
	rookOpenFileMg     = 20  // Rook bonus MG
	rookOpenFileEg     = 10  // Rook bonus EG
	rookSemiOpenFileMg = 10  // Rook semi bonus MG
	rookSemiOpenFileEg = 5   // Rook semi bonus EG
	doubledPawnMg      = -8  // Penalty per extra pawn on same file in MG
	doubledPawnEg      = -16 // Penalty per extra pawn on same file in EG
	bishopPairMg       = 13  // Bonus for bishop pair in MG
	bishopPairEg       = 26  // Bonus for bishop pair in EG
	isolatedPawnMg     = -6  // Penalty for iso pawn in MG
	isolatedPawnEg     = -10 // Penalty for iso pawn in EG
)

// Bit masks
const (
	maskSq    = 0x7F // 7-bit square mask
	maskPiece = 0xF  // 4-bit piece mask
	maskFlag  = 0x7  // 3-bit flag mask
)

// Move offset variables
var knightDirs = []int{-33, -31, -18, -14, 14, 18, 31, 33}
var bishopDirs = []int{-17, -15, 15, 17}
var rookDirs = []int{-16, -1, 1, 16}
var kingDirs = []int{-17, -16, -15, -1, 1, 15, 16, 17}

// Queens have same offsets as kings
var queenDirs = kingDirs

// Precomputed leaper tables
var knightTargets [128][]int
var kingTargets [128][]int
var pawnAttacks [2][128][]int

// Move encoding
func makeMoveEncoded(from, to, captured, promoted, flag int) Move {
	return Move(from | (to << 7) | (captured << 14) | (promoted << 18) | (flag << 22))
}

// Move decoding
func (m Move) from() int       { return int(m & maskSq) }
func (m Move) to() int         { return int((m >> 7) & maskSq) }
func (m Move) captured() int   { return int((m >> 14) & maskPiece) }
func (m Move) promoted() int   { return int((m >> 18) & maskPiece) }
func (m Move) flag() int       { return int((m >> 22) & maskFlag) }
func (m Move) isCapture() bool { return m.captured() != Empty }

// Move string
func (m Move) String() string {
	if m == NoMove {
		return "0000"
	}
	promo := ""
	if p := m.promoted(); p != Empty {
		promo = string("nbrq"[(p-WKnight)%6])
	}
	return sqName(m.from()) + sqName(m.to()) + promo
}

// UndoInfo for move unmaking
type UndoInfo struct {
	CastleRights int    // Castle
	EpSquare     int    // En passant
	HalfMove     int    // Halfmove
	Hash         uint64 // Board hash
}

// Complete board state
type Position struct {
	Board        [128]int  // Full board
	Side         int       // White or black
	CastleRights int       // Castle bitmask
	EpSquare     int       // En passant
	HalfMove     int       // Halfmove clock
	FullMove     int       // Full move counter
	KingSq       [2]int    // King square
	MgScore      [2]int    // Middlegame score
	EgScore      [2]int    // Endgame Score
	Phase        int       // Game phase
	Hash         uint64    // Hash of position
	History      []uint64  // History for repetition
	PieceCnt     [13]int   // Piece count
	PawnFileCnt  [2][8]int // Pawns per file
	RookFileCnt  [2][8]int // Rooks per file
}

// --------------------------
// --- 0x88 Board Helpers ---
// --------------------------

func sq88(rank, file int) int { return (rank << 4) | file } // Convert rank and file
func rankOf(sq int) int       { return sq >> 4 }            // Extract rank
func fileOf(sq int) int       { return sq & 7 }             // Extract file
func isSqValid(sq int) bool   { return (sq & 0x88) == 0 }   // Off-board check via nibble
func flipSq(sq int) int       { return sq ^ 0x70 }          // Mirror square vertically

// Piece colors per piece
var pieceColorTable = [13]int{-1, White, White, White, White, White, White, Black, Black, Black, Black, Black, Black}

// White or black piece?
func pieceColor(p int) int { return pieceColorTable[p] }

// Square name
func sqName(sq int) string {
	if !isSqValid(sq) {
		return "-"
	}
	return fmt.Sprintf("%c%c", 'a'+fileOf(sq), '1'+rankOf(sq))
}

// -----------------------
// --- Zobrist Hashing ---
// -----------------------

// Zobrist variables
var (
	zobristPieces [13][128]uint64 // Pieces
	zobristSide   uint64          // Side to move
	zobristCastle [16]uint64      // Castle
	zobristEp     [128]uint64     // En passant
)

// Initialize zobrist hashes
func initZobrist() {
	var seed uint64 = 69      // Random seed
	rand64 := func() uint64 { // XORShift
		seed ^= seed << 13
		seed ^= seed >> 7
		seed ^= seed << 17
		return seed
	}

	// Loop pieces
	for p := 0; p < 13; p++ {
		for sq := 0; sq < 128; sq++ {
			zobristPieces[p][sq] = rand64()
		}
	}
	// Get castle and en passant
	zobristSide = rand64()
	for i := 0; i < 16; i++ {
		zobristCastle[i] = rand64()
	}
	for sq := 0; sq < 128; sq++ {
		zobristEp[sq] = rand64()
	}
}

// ---------------------
// --- Leaper Tables ---
// ---------------------

func initLeaperTables() {
	// Loop over squares
	for sq := 0; sq < 128; sq++ {
		// Filter out illegals
		if !isSqValid(sq) {
			continue
		}
		// Knights
		for _, d := range knightDirs {
			// No illegals
			if to := sq + d; isSqValid(to) {
				knightTargets[sq] = append(knightTargets[sq], to)
			}
		}
		// Kings
		for _, d := range kingDirs {
			// No illegals here either
			if to := sq + d; isSqValid(to) {
				kingTargets[sq] = append(kingTargets[sq], to)
			}
		}
		// Pawns white
		for _, d := range []int{15, 17} {
			// Nor here
			if to := sq + d; isSqValid(to) {
				pawnAttacks[White][sq] = append(pawnAttacks[White][sq], to)
			}
		}
		// Pawns Black
		for _, d := range []int{-15, -17} {
			// Or here
			if to := sq + d; isSqValid(to) {
				pawnAttacks[Black][sq] = append(pawnAttacks[Black][sq], to)
			}
		}
	}
}

// -----------------------
// --- PSTs & Material ---
// -----------------------

// Middlegame piece square tables
var rawPstMg = [7][64]int{
	WPawn: {
		0, 0, 0, 0, 0, 0, 0, 0,
		-2, 2, 6, -18, -18, 6, 2, -2,
		-4, -3, -1, 10, 10, -1, -3, -4,
		-2, 1, 6, 22, 22, 6, 1, -2,
		4, 8, 16, 28, 28, 16, 8, 4,
		8, 14, 24, 32, 32, 24, 14, 8,
		45, 45, 45, 45, 45, 45, 45, 45,
		0, 0, 0, 0, 0, 0, 0, 0,
	},
	WKnight: {
		-55, -35, -25, -20, -20, -25, -35, -55,
		-35, -15, 2, 8, 8, 2, -15, -35,
		-20, 5, 18, 22, 22, 18, 5, -20,
		-15, 8, 22, 30, 30, 22, 8, -15,
		-15, 10, 22, 28, 28, 22, 10, -15,
		-20, 5, 15, 20, 20, 15, 5, -20,
		-35, -15, 0, 5, 5, 0, -15, -35,
		-55, -35, -25, -20, -20, -25, -35, -55,
	},
	WBishop: {
		-18, -8, -12, -6, -6, -12, -8, -18,
		-6, 12, 4, 6, 6, 4, 12, -6,
		-4, 8, 14, 12, 12, 14, 8, -4,
		-2, 4, 12, 20, 20, 12, 4, -2,
		-2, 6, 14, 18, 18, 14, 6, -2,
		-4, 8, 10, 14, 14, 10, 8, -4,
		-6, 8, 2, 4, 4, 2, 8, -6,
		-18, -12, -10, -8, -8, -10, -12, -18,
	},
	WRook: {
		-4, 0, 4, 8, 8, 4, 0, -4,
		-8, -2, 0, 2, 2, 0, -2, -8,
		-8, -2, 0, 2, 2, 0, -2, -8,
		-8, -2, 0, 4, 4, 0, -2, -8,
		-4, 0, 2, 6, 6, 2, 0, -4,
		-2, 2, 4, 6, 6, 4, 2, -2,
		12, 16, 18, 18, 18, 18, 16, 12,
		2, 4, 6, 10, 10, 6, 4, 2,
	},
	WQueen: {
		-22, -12, -8, -4, -4, -8, -12, -22,
		-12, -4, 0, 2, 2, 0, -4, -12,
		-8, 2, 4, 6, 6, 4, 2, -8,
		-4, 2, 6, 8, 8, 6, 2, -4,
		-2, 4, 6, 8, 8, 6, 4, -2,
		-8, 0, 4, 4, 4, 4, 0, -8,
		-12, -6, 0, 0, 0, 0, -6, -12,
		-22, -14, -10, -6, -6, -10, -14, -22,
	},
	WKing: {
		18, 28, 8, -8, -8, 8, 28, 18,
		16, 18, -4, -12, -12, -4, 18, 16,
		-14, -22, -24, -28, -28, -24, -22, -14,
		-24, -32, -36, -42, -42, -36, -32, -24,
		-34, -42, -44, -52, -52, -44, -42, -34,
		-38, -46, -48, -54, -54, -48, -46, -38,
		-44, -50, -52, -58, -58, -52, -50, -44,
		-48, -54, -56, -62, -62, -56, -54, -48,
	},
}

// Endgame piece square tables
var rawPstEg = [7][64]int{
	WPawn: {
		0, 0, 0, 0, 0, 0, 0, 0,
		-8, -4, -2, -2, -2, -2, -4, -8,
		-2, 2, 4, 6, 6, 4, 2, -2,
		6, 10, 14, 18, 18, 14, 10, 6,
		14, 20, 24, 28, 28, 24, 20, 14,
		28, 34, 38, 42, 42, 38, 34, 28,
		58, 60, 62, 62, 62, 62, 60, 58,
		0, 0, 0, 0, 0, 0, 0, 0,
	},
	WKnight: {
		-40, -25, -18, -15, -15, -18, -25, -40,
		-25, -10, 0, 5, 5, 0, -10, -25,
		-18, 0, 12, 16, 16, 12, 0, -18,
		-15, 5, 16, 22, 22, 16, 5, -15,
		-15, 5, 16, 22, 22, 16, 5, -15,
		-18, 0, 12, 16, 16, 12, 0, -18,
		-25, -10, 0, 5, 5, 0, -10, -25,
		-40, -25, -18, -15, -15, -18, -25, -40,
	},
	WBishop: {
		-16, -8, -6, -4, -4, -6, -8, -16,
		-8, 0, 4, 6, 6, 4, 0, -8,
		-6, 4, 10, 12, 12, 10, 4, -6,
		-4, 6, 12, 16, 16, 12, 6, -4,
		-4, 6, 12, 16, 16, 12, 6, -4,
		-6, 4, 10, 12, 12, 10, 4, -6,
		-8, 0, 4, 6, 6, 4, 0, -8,
		-16, -8, -6, -4, -4, -6, -8, -16,
	},
	WRook: {
		-6, -4, -2, 0, 0, -2, -4, -6,
		-2, 2, 4, 6, 6, 4, 2, -2,
		-2, 2, 4, 6, 6, 4, 2, -2,
		2, 4, 6, 8, 8, 6, 4, 2,
		4, 6, 8, 10, 10, 8, 6, 4,
		4, 6, 8, 10, 10, 8, 6, 4,
		18, 20, 22, 22, 22, 22, 20, 18,
		6, 8, 10, 12, 12, 10, 8, 6,
	},
	WQueen: {
		-14, -8, -4, 0, 0, -4, -8, -14,
		-8, 0, 4, 6, 6, 4, 0, -8,
		-4, 4, 10, 14, 14, 10, 4, -4,
		0, 6, 14, 20, 20, 14, 6, 0,
		0, 6, 14, 20, 20, 14, 6, 0,
		-4, 4, 10, 14, 14, 10, 4, -4,
		-8, 0, 4, 6, 6, 4, 0, -8,
		-14, -8, -4, 0, 0, -4, -8, -14,
	},
	WKing: {
		-45, -30, -20, -15, -15, -20, -30, -45,
		-25, -10, 5, 12, 12, 5, -10, -25,
		-15, 5, 18, 25, 25, 18, 5, -15,
		-10, 10, 25, 35, 35, 25, 10, -10,
		-10, 10, 25, 35, 35, 25, 10, -10,
		-15, 5, 18, 25, 25, 18, 5, -15,
		-25, -10, 5, 12, 12, 5, -10, -25,
		-45, -30, -20, -15, -15, -20, -30, -45,
	},
}

// Initialize piece square tables, add in materials and mirror for black
func initPST() {
	// Place material values into piece square tables
	for p := WPawn; p <= WKing; p++ {
		for r := 0; r < 8; r++ {
			for f := 0; f < 8; f++ {
				sq64 := r*8 + f
				sq88Idx := sq88(r, f)

				// White
				pstMg[p][sq88Idx] = rawPstMg[p][sq64] + materialMg[p]
				pstEg[p][sq88Idx] = rawPstEg[p][sq64] + materialEg[p]

				// Black
				// Mirror with flipsq
				bp := p + 6
				bsq88Idx := flipSq(sq88Idx)
				pstMg[bp][bsq88Idx] = rawPstMg[p][sq64] + materialMg[p]
				pstEg[bp][bsq88Idx] = rawPstEg[p][sq64] + materialEg[p]
			}
		}
	}
}

// ---------------------------
// --- Position Management ---
// ---------------------------

// CastleMask
var castleMask = [128]int{}

// Initialize castle mask so we don't need to compute inside move making
func initCastleMask() {
	// Loop through squares
	for i := 0; i < 128; i++ {
		castleMask[i] = WKCastle | WQCastle | BKCastle | BQCastle
	}
	// Set masks
	castleMask[sq88(0, 4)] &^= WKCastle | WQCastle
	castleMask[sq88(0, 0)] &^= WQCastle
	castleMask[sq88(0, 7)] &^= WKCastle
	castleMask[sq88(7, 4)] &^= BKCastle | BQCastle
	castleMask[sq88(7, 0)] &^= BQCastle
	castleMask[sq88(7, 7)] &^= BKCastle
}

// Add piece
func (pos *Position) addPiece(sq, p int) {
	pos.Board[sq] = p
	c := pieceColor(p)
	pos.MgScore[c] += pstMg[p][sq]
	pos.EgScore[c] += pstEg[p][sq]
	pos.Phase += phaseValues[p]
	pos.Hash ^= zobristPieces[p][sq]
	pos.PieceCnt[p]++
	switch p {
	case WPawn, BPawn:
		// Pawns (iso, doubled)
		pos.applyPawnFileChange(c, fileOf(sq), +1)
	case WRook, BRook:
		// Rooks (open, semi)
		pos.applyRookFileChange(c, fileOf(sq), +1)
	case WBishop, BBishop:
		// Bishop pair
		if pos.PieceCnt[p] == 2 {
			pos.MgScore[c] += bishopPairMg
			pos.EgScore[c] += bishopPairEg
		}
	case WKing, BKing:
		pos.KingSq[c] = sq
	}
}

// Remove piece
func (pos *Position) removePiece(sq int) {
	p := pos.Board[sq]
	c := pieceColor(p)
	pos.MgScore[c] -= pstMg[p][sq]
	pos.EgScore[c] -= pstEg[p][sq]
	pos.Phase -= phaseValues[p]
	pos.Hash ^= zobristPieces[p][sq]
	pos.PieceCnt[p]--
	switch p {
	case WPawn, BPawn:
		// Pawns (iso, doubled)
		pos.applyPawnFileChange(c, fileOf(sq), -1)
	case WRook, BRook:
		// Rooks (open, semi)
		pos.applyRookFileChange(c, fileOf(sq), -1)
	case WBishop, BBishop:
		// Bishop pair
		if pos.PieceCnt[p] == 1 {
			pos.MgScore[c] -= bishopPairMg
			pos.EgScore[c] -= bishopPairEg
		}
	}
	pos.Board[sq] = Empty
}

// Update piece
func (pos *Position) movePiece(from, to int) {
	p := pos.Board[from]
	c := pieceColor(p)
	pos.MgScore[c] -= pstMg[p][from]
	pos.EgScore[c] -= pstEg[p][from]
	pos.Hash ^= zobristPieces[p][from]

	pos.Board[from] = Empty // Set from square to empty
	pos.Board[to] = p       // Set to square to piece

	pos.MgScore[c] += pstMg[p][to]
	pos.EgScore[c] += pstEg[p][to]
	pos.Hash ^= zobristPieces[p][to]

	fFrom := fileOf(from)
	fTo := fileOf(to)
	switch p {
	case WPawn, BPawn:
		// Update pawn info
		if fFrom != fTo {
			pos.applyPawnFileChange(c, fFrom, -1)
			pos.applyPawnFileChange(c, fTo, +1)
		}
	case WRook, BRook:
		// Update rook info
		if fFrom != fTo {
			pos.applyRookFileChange(c, fFrom, -1)
			pos.applyRookFileChange(c, fTo, +1)
		}
	case WKing, BKing:
		pos.KingSq[c] = to
	}
}

// Parse given FEN string
func (pos *Position) parseFEN(fen string) {
	*pos = Position{EpSquare: -1, History: make([]uint64, 0, 256)}
	parts := strings.Split(fen, " ")
	// No parts?
	if len(parts) < 1 {
		return
	}

	// Piece placement
	rank, file := 7, 0
	for i := 0; i < len(parts[0]); i++ {
		c := parts[0][i]
		if c >= '1' && c <= '8' {
			file += int(c - '0')
			continue
		}
		if c == '/' {
			rank--
			file = 0
			continue
		}
		// Map pieces
		p := strings.IndexByte("PNBRQKpnbrqk", c) + 1
		// Check pieces, ranks and files
		if p != Empty && rank >= 0 && rank < 8 && file >= 0 && file < 8 {
			pos.addPiece(sq88(rank, file), p)
		}
		file++
	}

	// Side to move
	if len(parts) > 1 && parts[1] == "b" {
		pos.Side = Black
		pos.Hash ^= zobristSide
	}

	// Castling rights
	if len(parts) > 2 {
		for i := 0; i < len(parts[2]); i++ {
			switch parts[2][i] {
			case 'K':
				pos.CastleRights |= WKCastle
			case 'Q':
				pos.CastleRights |= WQCastle
			case 'k':
				pos.CastleRights |= BKCastle
			case 'q':
				pos.CastleRights |= BQCastle
			}
		}
	}
	pos.Hash ^= zobristCastle[pos.CastleRights]

	// En passant
	if len(parts) > 3 && parts[3] != "-" && len(parts[3]) >= 2 {
		f := int(parts[3][0] - 'a')
		r := int(parts[3][1] - '1')
		if f >= 0 && f < 8 && (r == 2 || r == 5) {
			pos.EpSquare = sq88(r, f)
			pos.Hash ^= zobristEp[pos.EpSquare]
		}
	}

	// Halfmove and fullmove
	if len(parts) > 4 {
		pos.HalfMove, _ = strconv.Atoi(parts[4])
	}
	if len(parts) > 5 {
		pos.FullMove, _ = strconv.Atoi(parts[5])
	}
	if pos.FullMove == 0 {
		pos.FullMove = 1
	}
}

// Starting position FEN
const startPos = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

// -----------------------
// --- Move Generation ---
// -----------------------

// Generate sliding moves
func (pos *Position) genSlider(moves []Move, sq int, dirs []int, enemy int, capturesOnly bool) []Move {
	for _, d := range dirs {
		for to := sq + d; isSqValid(to); to += d {
			capP := pos.Board[to]
			if capP == Empty {
				if !capturesOnly {
					moves = append(moves, makeMoveEncoded(sq, to, Empty, Empty, FlagNone))
				}
			} else {
				if pieceColor(capP) == enemy {
					moves = append(moves, makeMoveEncoded(sq, to, capP, Empty, FlagNone))
				}
				break
			}
		}
	}
	return moves
}

// Generate leaper moves
func (pos *Position) genLeaper(moves []Move, sq int, targets []int, enemy int, capturesOnly bool) []Move {
	for _, to := range targets {
		capP := pos.Board[to]
		if capP == Empty {
			if !capturesOnly {
				moves = append(moves, makeMoveEncoded(sq, to, Empty, Empty, FlagNone))
			}
		} else if pieceColor(capP) == enemy {
			moves = append(moves, makeMoveEncoded(sq, to, capP, Empty, FlagNone))
		}
	}
	return moves
}

// Generate all moves
func (pos *Position) generateMoves(buf []Move, capturesOnly bool) []Move {
	moves := buf[:0]

	side := pos.Side
	enemy := side ^ 1

	// Loop over squares
	for sq := 0; sq < 128; sq++ {
		if !isSqValid(sq) {
			continue
		}
		p := pos.Board[sq]
		if p == Empty || pieceColor(p) != side {
			continue
		}

		// Pawns
		switch p {
		case WPawn, BPawn:
			push := 16 - side*32    // +16 White, -16 Black
			promoRank := 6 - side*5 // rank 6 White, rank 1 Black
			startRank := 1 + side*5 // rank 1 White, rank 6 Black
			if !capturesOnly && isSqValid(sq+push) && pos.Board[sq+push] == Empty {
				if rankOf(sq) == promoRank {
					for prom := WQueen + side*6; prom >= WKnight+side*6; prom-- {
						moves = append(moves, makeMoveEncoded(sq, sq+push, Empty, prom, FlagPromotion))
					}
				} else {
					moves = append(moves, makeMoveEncoded(sq, sq+push, Empty, Empty, FlagNone))
					if rankOf(sq) == startRank && pos.Board[sq+push*2] == Empty {
						moves = append(moves, makeMoveEncoded(sq, sq+push*2, Empty, Empty, FlagDoublePush))
					}
				}
			}
			for _, to := range pawnAttacks[side][sq] {
				capP := pos.Board[to]
				if capP != Empty && pieceColor(capP) == enemy {
					if rankOf(sq) == promoRank {
						for prom := WQueen + side*6; prom >= WKnight+side*6; prom-- {
							moves = append(moves, makeMoveEncoded(sq, to, capP, prom, FlagPromotion))
						}
					} else {
						moves = append(moves, makeMoveEncoded(sq, to, capP, Empty, FlagNone))
					}
				} else if to == pos.EpSquare {
					moves = append(moves, makeMoveEncoded(sq, to, BPawn-side*6, Empty, FlagEP))
				}
			}
		case WKnight, BKnight:
			moves = pos.genLeaper(moves, sq, knightTargets[sq], enemy, capturesOnly)
		case WBishop, BBishop:
			moves = pos.genSlider(moves, sq, bishopDirs, enemy, capturesOnly)
		case WRook, BRook:
			moves = pos.genSlider(moves, sq, rookDirs, enemy, capturesOnly)
		case WQueen, BQueen:
			moves = pos.genSlider(moves, sq, queenDirs, enemy, capturesOnly)
		case WKing, BKing:
			moves = pos.genLeaper(moves, sq, kingTargets[sq], enemy, capturesOnly)

			// Castling
			if !capturesOnly {
				r := side * 7
				if pos.CastleRights&(WKCastle<<(side*2)) != 0 && pos.Board[sq88(r, 5)] == Empty && pos.Board[sq88(r, 6)] == Empty {
					if !pos.isSquareAttacked(sq88(r, 4), enemy) && !pos.isSquareAttacked(sq88(r, 5), enemy) {
						moves = append(moves, makeMoveEncoded(sq88(r, 4), sq88(r, 6), Empty, Empty, FlagCastle))
					}
				}
				if pos.CastleRights&(WQCastle<<(side*2)) != 0 && pos.Board[sq88(r, 3)] == Empty && pos.Board[sq88(r, 2)] == Empty && pos.Board[sq88(r, 1)] == Empty {
					if !pos.isSquareAttacked(sq88(r, 4), enemy) && !pos.isSquareAttacked(sq88(r, 3), enemy) {
						moves = append(moves, makeMoveEncoded(sq88(r, 4), sq88(r, 2), Empty, Empty, FlagCastle))
					}
				}
			}
		}
	}
	return moves
}

// Is square attacked?
func (pos *Position) isSquareAttacked(sq int, bySide int) bool {
	off := bySide * 6
	// Pawns in reverse
	for _, from := range pawnAttacks[bySide^1][sq] {
		if pos.Board[from] == WPawn+off {
			return true
		}
	}
	// Knights
	for _, to := range knightTargets[sq] {
		if pos.Board[to] == WKnight+off {
			return true
		}
	}
	// Kings
	for _, to := range kingTargets[sq] {
		if pos.Board[to] == WKing+off {
			return true
		}
	}
	// Bishops & queens
	for _, d := range bishopDirs {
		for to := sq + d; isSqValid(to); to += d {
			if p := pos.Board[to]; p != Empty {
				if p == WBishop+off || p == WQueen+off {
					return true
				}
				break
			}
		}
	}
	// Rooks & queens
	for _, d := range rookDirs {
		for to := sq + d; isSqValid(to); to += d {
			if p := pos.Board[to]; p != Empty {
				if p == WRook+off || p == WQueen+off {
					return true
				}
				break
			}
		}
	}
	return false
}

// --------------------------
// --- Make / Unmake Move ---
// --------------------------
// NOTE: If swap to Neural networks rewrite this to copy & make

// Make move on board
func (pos *Position) makeMove(m Move, undo *UndoInfo) bool {
	undo.CastleRights = pos.CastleRights
	undo.EpSquare = pos.EpSquare
	undo.HalfMove = pos.HalfMove
	undo.Hash = pos.Hash

	from := m.from()
	to := m.to()
	capP := m.captured()
	prom := m.promoted()
	flag := m.flag()

	pos.History = append(pos.History, pos.Hash)

	if pos.EpSquare != -1 {
		pos.Hash ^= zobristEp[pos.EpSquare]
		pos.EpSquare = -1
	}
	pos.Hash ^= zobristCastle[pos.CastleRights]

	if capP != Empty && flag != FlagEP {
		pos.removePiece(to)
	}

	pos.movePiece(from, to)

	if prom != Empty {
		pos.removePiece(to)
		pos.addPiece(to, prom)
	}

	p := pos.Board[to]

	// Reset halfmove clock if pawn, capture or promotion
	if p == WPawn || p == BPawn || capP != Empty || prom != Empty {
		pos.HalfMove = 0
	} else {
		pos.HalfMove++
	}

	switch flag {
	case FlagDoublePush:
		pos.EpSquare = from + 16 - pos.Side*32
		pos.Hash ^= zobristEp[pos.EpSquare]
	case FlagEP:
		pos.removePiece(to - 16 + pos.Side*32)
	case FlagCastle:
		r := rankOf(to)
		f := fileOf(to) >> 2
		pos.movePiece(sq88(r, f*7), sq88(r, f*2+3))
	}

	pos.CastleRights &= castleMask[from] & castleMask[to]
	pos.Hash ^= zobristCastle[pos.CastleRights]

	pos.Side ^= 1
	pos.Hash ^= zobristSide
	if pos.Side == White {
		pos.FullMove++
	}

	// Legality check
	kingSq := pos.KingSq[pos.Side^1]
	if pos.isSquareAttacked(kingSq, pos.Side) {
		pos.unmakeMove(m, undo)
		return false
	}

	return true
}

func (pos *Position) unmakeMove(m Move, undo *UndoInfo) {
	if pos.Side == White {
		pos.FullMove--
	}
	pos.Side ^= 1

	from := m.from()
	to := m.to()
	capP := m.captured()
	prom := m.promoted()
	flag := m.flag()

	if prom != Empty {
		pos.removePiece(to)
		pos.addPiece(to, WPawn+pos.Side*6)
	}

	pos.movePiece(to, from)

	if capP != Empty && flag != FlagEP {
		pos.addPiece(to, capP)
	}

	switch flag {
	case FlagEP:
		pos.addPiece(to-16+pos.Side*32, BPawn-pos.Side*6)
	case FlagCastle:
		r := rankOf(to)
		f := fileOf(to) >> 2
		pos.movePiece(sq88(r, f*2+3), sq88(r, f*7))
	}

	pos.CastleRights = undo.CastleRights
	pos.EpSquare = undo.EpSquare
	pos.HalfMove = undo.HalfMove
	pos.Hash = undo.Hash

	pos.History = pos.History[:len(pos.History)-1]
}

// Used for null move pruning conditions
// Checks if we have non pawn material phase
func (pos *Position) nonPawnPhase(side int) int {
	return pos.PieceCnt[WKnight+side*6]*phaseValues[WKnight] +
		pos.PieceCnt[WBishop+side*6]*phaseValues[WBishop] +
		pos.PieceCnt[WRook+side*6]*phaseValues[WRook] +
		pos.PieceCnt[WQueen+side*6]*phaseValues[WQueen]
}

// Make a null move
func (pos *Position) makeNullMove(undo *UndoInfo) {
	undo.EpSquare = pos.EpSquare
	undo.Hash = pos.Hash
	if pos.EpSquare != -1 {
		pos.Hash ^= zobristEp[pos.EpSquare]
		pos.EpSquare = -1
	}
	pos.Side ^= 1
	pos.Hash ^= zobristSide
}

// Unmake a null move
func (pos *Position) unmakeNullMove(undo *UndoInfo) {
	pos.Side ^= 1
	pos.EpSquare = undo.EpSquare
	pos.Hash = undo.Hash
}

// Is position a draw?
func (pos *Position) isDraw() bool {
	// 50 move rule
	if pos.HalfMove >= 100 {
		return true
	}

	// Twofold repetitions
	for i := len(pos.History) - 2; i >= 0 && i >= len(pos.History)-pos.HalfMove; i -= 2 {
		if pos.History[i] == pos.Hash {
			return true
		}
	}

	// Insufficient material
	if pos.PieceCnt[WPawn]+pos.PieceCnt[BPawn]+pos.PieceCnt[WRook]+pos.PieceCnt[BRook]+pos.PieceCnt[WQueen]+pos.PieceCnt[BQueen] > 0 {
		return false
	}
	wMinor := pos.PieceCnt[WKnight] + pos.PieceCnt[WBishop]
	bMinor := pos.PieceCnt[BKnight] + pos.PieceCnt[BBishop]
	return wMinor+bMinor <= 2 && wMinor <= 1 && bMinor <= 1
}

// ------------------
// --- Evaluation ---
// ------------------

// Rook bonus helper
func (pos *Position) rookFileBonus(side, f int) (mg, eg int) {
	// Is there a own pawn in the way?
	if pos.PawnFileCnt[side][f] != 0 {
		// Pawn in the way
		return 0, 0
	}
	// Is there enemy pawn in the way?
	if pos.PawnFileCnt[side^1][f] == 0 {
		// No pawn in the way
		return rookOpenFileMg, rookOpenFileEg
	}
	// No own pawn, but not empty for enemy pawn
	// Enemy pawn in the way
	return rookSemiOpenFileMg, rookSemiOpenFileEg
}

// Isolated pawn helper
func (pos *Position) isoCount(c, f int) int {
	// Is there a pawn?
	if f < 0 || f > 7 || pos.PawnFileCnt[c][f] == 0 {
		return 0
	}
	// Check neighbor files
	if (f == 0 || pos.PawnFileCnt[c][f-1] == 0) && (f == 7 || pos.PawnFileCnt[c][f+1] == 0) {
		// Neither file has neighbor pawn
		return pos.PawnFileCnt[c][f]
	}
	// Has neighbor pawn(s)
	return 0
}

// Apply changes to pawn file
func (pos *Position) applyPawnFileChange(c, f, delta int) {
	// Compute values, old and new
	oldWMg, oldWEg := pos.rookFileBonus(White, f)
	oldBMg, oldBEg := pos.rookFileBonus(Black, f)
	oldIso := pos.isoCount(c, f-1) + pos.isoCount(c, f) + pos.isoCount(c, f+1)
	oldCnt := pos.PawnFileCnt[c][f]
	pos.PawnFileCnt[c][f] += delta
	newCnt := pos.PawnFileCnt[c][f]
	newWMg, newWEg := pos.rookFileBonus(White, f)
	newBMg, newBEg := pos.rookFileBonus(Black, f)
	nW := pos.RookFileCnt[White][f]
	nB := pos.RookFileCnt[Black][f]
	pos.MgScore[White] += nW * (newWMg - oldWMg)
	pos.EgScore[White] += nW * (newWEg - oldWEg)
	pos.MgScore[Black] += nB * (newBMg - oldBMg)
	pos.EgScore[Black] += nB * (newBEg - oldBEg)

	// Update doubled pawn score
	diff := max(0, newCnt-1) - max(0, oldCnt-1)
	pos.MgScore[c] += diff * doubledPawnMg
	pos.EgScore[c] += diff * doubledPawnEg

	// Update iso pawn score
	newIso := pos.isoCount(c, f-1) + pos.isoCount(c, f) + pos.isoCount(c, f+1)
	pos.MgScore[c] += (newIso - oldIso) * isolatedPawnMg
	pos.EgScore[c] += (newIso - oldIso) * isolatedPawnEg
}

// Apply changes to rook file
func (pos *Position) applyRookFileChange(side, f, delta int) {
	mg, eg := pos.rookFileBonus(side, f)
	pos.RookFileCnt[side][f] += delta
	pos.MgScore[side] += delta * mg
	pos.EgScore[side] += delta * eg
}

// Evaluate position
// Return from side to move perspective
func (pos *Position) evaluate() int {
	// Check current phase
	phase := min(pos.Phase, maxPhase)

	// Compute evaluations from incrementally maintained values
	mg := pos.MgScore[pos.Side] - pos.MgScore[pos.Side^1]
	eg := pos.EgScore[pos.Side] - pos.EgScore[pos.Side^1]

	// Interpolate MG & EG
	score := (mg*phase + eg*(maxPhase-phase)) / maxPhase
	// Return and add in tempo
	return score + tempoBonus
}

// ---------------------
// --- Move Ordering ---
// ---------------------

// Assign ordering scores
func scoreMove(pos *Position, m Move, pvMove, ttMove Move, ply, side int) int {
	// Previous best move?
	if m == pvMove {
		return scorePVMove
	}
	// TT move?
	if m == ttMove {
		return scorePVMove - 1
	}
	// Capture?
	if m.isCapture() {
		victim := m.captured()
		attacker := pos.Board[m.from()]
		return mvvLvaVal[victim]*mvvMultiplier - mvvLvaVal[attacker] + scoreCapBase
	}
	// Killer A?
	if m == killers[ply][0] {
		return scoreKillerA
	}
	// Killer B?
	if m == killers[ply][1] {
		return scoreKillerB
	}
	// History score
	return history[side][m.from()][m.to()]
}

// Sort moves based on scores given
func sortMoves(moves []Move, scores []int, current int) {
	bestIdx := current
	bestScore := scores[current]
	for i := current + 1; i < len(moves); i++ {
		if scores[i] > bestScore {
			bestScore = scores[i]
			bestIdx = i
		}
	}
	moves[current], moves[bestIdx] = moves[bestIdx], moves[current]
	scores[current], scores[bestIdx] = scores[bestIdx], scores[current]
}

// ---------------------------
// --- Transposition Table ---
// ---------------------------

// Initialize tt
func initTT(sizeMB int) {
	// Clamp size
	sizeMB = min(ttMaxMB, max(ttMinMB, sizeMB))
	// Calculate entries
	entryBytes := int(unsafe.Sizeof(TTEntry{}))
	nEntries := (sizeMB * 1024 * 1024) / entryBytes
	size := 1
	// Round down to power of two
	for size*2 <= nEntries {
		size *= 2
	}
	tt = make([]TTEntry, size)
	ttMask = uint64(size - 1)
}

// Wipe all entries
func clearTT() {
	clear(tt)
}

// Adjust scores if mate scores (storing)
func ttAdjustStore(score, ply int) int16 {
	if score > mateThreshold {
		return int16(score + ply)
	}
	if score < -mateThreshold {
		return int16(score - ply)
	}
	return int16(score)
}

// Adjust scores if mate scores (retrieving)
func ttAdjustRetrieve(stored int16, ply int) int {
	score := int(stored)
	if score > mateThreshold {
		return score - ply
	}
	if score < -mateThreshold {
		return score + ply
	}
	return score
}

// Check if TT has anything useful
func ttProbe(hash uint64) (TTEntry, bool) {
	e := tt[hash&ttMask]
	return e, e.Hash == hash
}

// Store to TT, always replace
func ttStore(hash uint64, move Move, score, depth, ply int, flag uint8) {
	idx := hash & ttMask
	tt[idx] = TTEntry{
		Hash:  hash,
		Move:  move,
		Score: ttAdjustStore(score, ply),
		Depth: int8(depth),
		Flag:  flag,
	}
}

// Format score if mate
func formatScore(score int) string {
	if score > mateThreshold {
		plies := mateScore - score
		return fmt.Sprintf("mate %d", (plies+1)/2)
	}
	if score < -mateThreshold {
		plies := mateScore + score
		return fmt.Sprintf("mate -%d", (plies+1)/2)
	}
	return fmt.Sprintf("cp %d", score)
}

// Age history, halve all scores
// TODO: Test if / 4 better
func ageHistory() {
	for side := 0; side < 2; side++ {
		for from := 0; from < 128; from++ {
			for to := 0; to < 128; to++ {
				history[side][from][to] /= 2
			}
		}
	}
}

// Clear history
// Go zero initializes
func clearHistory() {
	history = [2][128][128]int{}
}

// Update history scores
func updateHistory(side int, moves []Move, cutoffIdx, depth int) {
	bonus := depth * depth
	m := moves[cutoffIdx]
	h := &history[side][m.from()][m.to()]
	*h = min(*h+bonus, scoreCapBase-1)
	for j := 0; j < cutoffIdx; j++ {
		if !moves[j].isCapture() {
			hp := &history[side][moves[j].from()][moves[j].to()]
			*hp = max(*hp-bonus, -(scoreCapBase - 1))
		}
	}
}

// --------------
// --- Search ---
// --------------

// Increment node counter, check time, return true if search should stop
func tickNodes() bool {
	if nodes&timeCheckMask == 0 {
		checkTime()
	}
	nodes++
	return abortSearch.Load()
}

// Check time limit
func checkTime() {
	if useTime && time.Now().After(stopTime) {
		abortSearch.Store(true)
	}
}

// Search until no captures
func quiescence(pos *Position, alpha, beta int) int {
	if tickNodes() {
		return 0
	}
	// Check draws
	if pos.isDraw() {
		return 0
	}

	// Stand pat
	standPat := pos.evaluate()
	if standPat >= beta {
		return standPat
	}
	bestScore := standPat
	if standPat > alpha {
		alpha = standPat
	}

	// Generate only captures for quiescence
	var moveBuf [maxMoves]Move
	moves := pos.generateMoves(moveBuf[:0], true)
	var scoreBuf [maxMoves]int
	scores := scoreBuf[:len(moves)]
	for i, m := range moves {
		scores[i] = scoreMove(pos, m, NoMove, NoMove, 0, pos.Side)
	}

	var undo UndoInfo
	for i := 0; i < len(moves); i++ {
		sortMoves(moves, scores, i)

		if !pos.makeMove(moves[i], &undo) {
			continue
		}
		// Recurse
		score := -quiescence(pos, -beta, -alpha)
		pos.unmakeMove(moves[i], &undo)

		// Check if search should stop inside loop
		if abortSearch.Load() {
			return 0
		}

		if score > bestScore {
			bestScore = score
			if score > alpha {
				alpha = score
			}
			if score >= beta {
				break
			}
		}
	}
	return bestScore
}

// Search through moves with negamax
func alphaBeta(pos *Position, alpha, beta, depth, ply int, pvMove Move, pvLine *[]Move) int {
	// Clear PV line
	*pvLine = nil

	if tickNodes() {
		return 0
	}

	// Exceed max depth?
	if ply >= maxSearchDepth {
		return pos.evaluate()
	}

	// Check draws
	if ply > 0 && pos.isDraw() {
		return 0
	}

	// Probe TT
	var ttMove Move
	if entry, hit := ttProbe(pos.Hash); hit {
		ttMove = entry.Move
		if ply > 0 && int(entry.Depth) >= depth {
			s := ttAdjustRetrieve(entry.Score, ply)
			switch entry.Flag {
			case ttExact:
				return s
			case ttBeta:
				if s >= beta {
					return s
				}
			case ttAlpha:
				if s <= alpha {
					return s
				}
			}
		}
	}

	// Extend if in check, stops going to quiescence in check
	inCheck := pos.isSquareAttacked(pos.KingSq[pos.Side], pos.Side^1)
	if inCheck {
		depth++
	}

	// Null move pruning
	if !inCheck && depth >= 3 && ply > 0 && beta < mateThreshold && pos.nonPawnPhase(pos.Side) > 0 {
		var nullUndo UndoInfo
		pos.makeNullMove(&nullUndo)
		var nullPv []Move
		score := -alphaBeta(pos, -beta, -beta+1, depth-1-NullMoveReduction, ply+1, NoMove, &nullPv)
		pos.unmakeNullMove(&nullUndo)
		// Check if search should stop
		if abortSearch.Load() {
			return 0
		}
		if score >= beta {
			// Dont propagate an unproven mate score from a null move.
			if score >= mateThreshold {
				return beta
			}
			return score
		}
	}

	// Quiescence if 0 or below
	if depth <= 0 {
		return quiescence(pos, alpha, beta)
	}

	// Generate moves
	var moveBuf [maxMoves]Move
	moves := pos.generateMoves(moveBuf[:0], false)
	var scoreBuf [maxMoves]int
	scores := scoreBuf[:len(moves)]
	// Order moves
	for i, m := range moves {
		scores[i] = scoreMove(pos, m, pvMove, ttMove, ply, pos.Side)
	}

	var undo UndoInfo
	legalMoves := 0
	var bestMove Move
	bestScore := -infScore
	origAlpha := alpha

	for i := 0; i < len(moves); i++ {
		sortMoves(moves, scores, i)
		m := moves[i]
		mover := pos.Side

		if !pos.makeMove(m, &undo) {
			continue
		}
		legalMoves++

		// Principal Variation Search.
		var line []Move
		var score int
		if legalMoves == 1 {
			score = -alphaBeta(pos, -beta, -alpha, depth-1, ply+1, NoMove, &line)
		} else {
			// Late move reductions
			nd := depth - 1
			if legalMoves > lmrMinLegalMoves && depth >= lmrMinDepth && !m.isCapture() && m.promoted() == Empty && !inCheck {
				R := 1
				// Depth higher than or equal to 6?
				if depth >= lmrDepthBumpAt {
					R++
				}
				// Moves higher than or 6?
				if legalMoves >= lmrLegalBumpAt {
					R++
				}
				nd = max(depth-1-R, 1)
			}
			// Null-window scout
			score = -alphaBeta(pos, -alpha-1, -alpha, nd, ply+1, NoMove, &line)
			// Beat alpha while reduced? Re-search full depth
			if score > alpha && nd < depth-1 {
				line = nil
				score = -alphaBeta(pos, -alpha-1, -alpha, depth-1, ply+1, NoMove, &line)
			}
			// Beat alpha at a PV node? Re-search full window
			if score > alpha && score < beta {
				line = nil
				score = -alphaBeta(pos, -beta, -alpha, depth-1, ply+1, NoMove, &line)
			}
		}
		pos.unmakeMove(m, &undo)

		// Check if should stop
		if abortSearch.Load() {
			return 0
		}

		if score > bestScore {
			bestScore = score
			bestMove = m
			if score > alpha {
				alpha = score
				if score < beta {
					// New principal variation at a PV node
					*pvLine = append([]Move{m}, line...)
				} else {
					// Fail-high beta cutoff
					if !m.isCapture() {
						updateHistory(mover, moves, i, depth)
						if killers[ply][0] != m {
							killers[ply][1] = killers[ply][0]
							killers[ply][0] = m
						}
					}
					break
				}
			}
		}
	}

	// 0 moves, so either draw or mate
	if legalMoves == 0 {
		if inCheck {
			score := -mateScore + ply
			ttStore(pos.Hash, NoMove, score, depth, ply, ttExact)
			return score
		}
		ttStore(pos.Hash, NoMove, 0, depth, ply, ttExact)
		return 0 // Stalemate
	}

	// TT flag
	flag := ttAlpha
	if bestScore >= beta {
		flag = ttBeta
	} else if bestScore > origAlpha {
		flag = ttExact
	}
	ttStore(pos.Hash, bestMove, bestScore, depth, ply, flag)
	return bestScore
}

// Search through depths
func search(pos *Position, maxDepth int, maxNodes uint64) Move {
	nodes = 0
	abortSearch.Store(false)
	// Age history values
	ageHistory()
	// Clear killers (they are local to one search)
	killers = [maxSearchDepth][2]Move{}

	var bestMove Move
	var globalPv []Move
	startTime := time.Now()
	prevScore := pos.evaluate()

	// Loop through depths
	for depth := 1; depth <= maxDepth; depth++ {
		pvMove := NoMove
		if len(globalPv) > 0 {
			pvMove = globalPv[0]
		}

		// Aspiration window
		alpha, beta, delta := -infScore, infScore, aspirationDelta
		if depth >= 4 {
			alpha, beta = prevScore-delta, prevScore+delta
		}
		var pv []Move
		var score int

		for retries := 0; retries < aspirationMaxRetries; retries++ {
			pv = nil
			score = alphaBeta(pos, alpha, beta, depth, 0, pvMove, &pv)
			if abortSearch.Load() || (score > alpha && score < beta) {
				break
			}
			if score <= alpha {
				alpha = max(alpha-delta, -infScore)
			} else {
				beta = min(beta+delta, infScore)
			}
			// Keep growing window by delta
			delta *= 2
		}
		// All retries done, just go full window
		if !abortSearch.Load() && (score <= alpha || score >= beta) {
			pv = nil
			score = alphaBeta(pos, -infScore, infScore, depth, 0, pvMove, &pv)
		}

		// Check if should stop
		if abortSearch.Load() {
			break
		}
		if len(pv) > 0 {
			bestMove = pv[0]
			globalPv = pv
		}
		prevScore = score

		// Print info
		elapsed := max(time.Since(startTime).Milliseconds(), 1)
		nps := (nodes * 1000) / uint64(elapsed)

		fmt.Printf("info depth %d score %s nodes %d nps %d time %d pv", depth, formatScore(score), nodes, nps, elapsed)
		for _, m := range globalPv {
			fmt.Printf(" %s", m.String())
		}
		fmt.Println()

		// Node limit?
		if maxNodes > 0 && nodes >= maxNodes {
			break
		}

		// Have we used our allocated time?
		if useTime && time.Now().After(stopTime) {
			break
		}
	}

	fmt.Printf("bestmove %s\n", bestMove.String())
	return bestMove
}

// -----------------------
// --- Time Management ---
// -----------------------

// Calculate time to search
func calcTime(wtime, btime, winc, binc, movestogo int, side int) int {
	timeLeft := wtime + side*(btime-wtime)
	inc := winc + side*(binc-winc)
	div := MovesLeft
	// No movestogo?
	if movestogo > 0 {
		div = movestogo
	}
	return max(timeLeft/div+inc/2, 1)
}

// --------------------
// --- UCI Protocol ---
// --------------------

func uciLoop() {
	scanner := bufio.NewScanner(os.Stdin)
	var pos Position
	pos.parseFEN(startPos)

	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.Fields(line)
		if len(parts) == 0 {
			continue
		}

		cmd := parts[0]
		switch cmd {
		case "uci":
			fmt.Println("id name PumpKing")
			fmt.Println("id author Otto Laukkanen")
			fmt.Printf("option name Hash type spin default %d min %d max %d\n", ttDefMB, ttMinMB, ttMaxMB)
			fmt.Println("uciok")
		case "isready":
			waitForSearch()
			fmt.Println("readyok")
		case "ucinewgame":
			waitForSearch()
			pos.parseFEN(startPos)
			clearTT()
			clearHistory()
		case "setoption":
			waitForSearch()
			if len(parts) >= 5 && strings.EqualFold(parts[2], "Hash") {
				mb, err := strconv.Atoi(parts[4])
				if err == nil {
					initTT(mb)
				}
			}
		case "position":
			waitForSearch()
			idx := 1
			if idx < len(parts) && parts[idx] == "startpos" {
				pos.parseFEN(startPos)
				idx++
			} else if idx < len(parts) && parts[idx] == "fen" {
				fenParts := []string{}
				idx++
				for idx < len(parts) && parts[idx] != "moves" {
					fenParts = append(fenParts, parts[idx])
					idx++
				}
				pos.parseFEN(strings.Join(fenParts, " "))
			}
			if idx < len(parts) && parts[idx] == "moves" {
				idx++
				for idx < len(parts) {
					moveStr := parts[idx]
					var moveBuf [maxMoves]Move
					moves := pos.generateMoves(moveBuf[:0], false)
					var undo UndoInfo
					found := false
					for _, m := range moves {
						if m.String() == moveStr {
							found = true
							if !pos.makeMove(m, &undo) {
								fmt.Printf("info string error: illegal move %s\n", moveStr)
								idx = len(parts)
							}
							break
						}
					}
					if !found {
						fmt.Printf("info string error: unknown move %s\n", moveStr)
						break
					}
					idx++
				}
			}
		case "go":
			wtime, btime, winc, binc, movestogo, depth, nodesLimit, movetime := 0, 0, 0, 0, 0, maxSearchDepth, uint64(0), 0
			for i := 1; i+1 < len(parts); i += 2 {
				switch parts[i] {
				case "wtime":
					wtime, _ = strconv.Atoi(parts[i+1])
				case "btime":
					btime, _ = strconv.Atoi(parts[i+1])
				case "winc":
					winc, _ = strconv.Atoi(parts[i+1])
				case "binc":
					binc, _ = strconv.Atoi(parts[i+1])
				case "movestogo":
					movestogo, _ = strconv.Atoi(parts[i+1])
				case "depth":
					depth, _ = strconv.Atoi(parts[i+1])
				case "nodes":
					n, _ := strconv.ParseUint(parts[i+1], 10, 64)
					nodesLimit = n
				case "movetime":
					movetime, _ = strconv.Atoi(parts[i+1])
				}
			}

			// Wait for in flight search to finish
			waitForSearch()

			useTime = false
			if movetime > 0 {
				useTime = true
				adj := max(movetime-movetimeSafetyMs, 1)
				stopTime = time.Now().Add(time.Duration(adj) * time.Millisecond)
			} else if wtime > 0 || btime > 0 {
				useTime = true
				allocated := calcTime(wtime, btime, winc, binc, movestogo, pos.Side)
				stopTime = time.Now().Add(time.Duration(allocated) * time.Millisecond)
			}

			abortSearch.Store(false)
			searchDone = make(chan struct{})
			go func(done chan struct{}, depth int, nodesLimit uint64) {
				defer close(done)
				search(&pos, depth, nodesLimit)
			}(searchDone, depth, nodesLimit)
		case "stop":
			abortSearch.Store(true)
		case "quit":
			waitForSearch()
			return
		}
	}
}

// ------------
// --- Main ---
// ------------

// Initialize everything
func init() {
	initZobrist()
	initPST()
	initCastleMask()
	initLeaperTables()
	initTT(ttDefMB)
}

func main() {
	uciLoop()
}
