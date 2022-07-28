figure_height="1.7in"
figure_width="3.47in"

set terminal epslatex size figure_width,figure_height

set datafile separator ","
set key outside above
set key reverse Left

set key samplen 1
set key width -6
set ytics 10

#set lmargin at screen 0.135
set rmargin at screen 0.98
#set tmargin at screen 0.98
#set bmargin at screen 0.2

set logscale y

set ylabel "R. throughput"
set xlabel "Number of dimensions"

set output "layout_CLANG_znver2.tex"
plot "output_CLANG_znver2.csv" u 1:2 every ::1 w linespoints pt 1 lc "#b6251c" t "Morton (naive)",\
     "" u 1:3 every ::1 w linespoints pt 1 lc "#149b52" t "Morton (\\texttt{PDEP})",\
     "" u 1:6 every ::1 w linespoints pt 1 lc "#52399d" t "Pitched (naive)",\
     "" u 1:5 every ::1 w linespoints pt 1 lc "#FFCA3A" t "Pitched (precomp.)"

set output "layout_GCC_znver2.tex"
plot "output_GCC_znver2.csv" u 1:2 every ::1 w linespoints pt 1 lc "#b6251c" t "Morton (naive)",\
     "" u 1:3 every ::1 w linespoints pt 1 lc "#149b52" t "Morton (\\texttt{PDEP})",\
     "" u 1:6 every ::1 w linespoints pt 1 lc "#52399d" t "Pitched (naive)",\
     "" u 1:5 every ::1 w linespoints pt 1 lc "#FFCA3A" t "Pitched (precomp.)"

set output "layout_CLANG_haswell.tex"
plot "output_CLANG_haswell.csv" u 1:2 every ::1 w linespoints pt 1 lc "#b6251c" t "Morton (naive)",\
     "" u 1:3 every ::1 w linespoints pt 1 lc "#149b52" t "Morton (\\texttt{PDEP})",\
     "" u 1:6 every ::1 w linespoints pt 1 lc "#52399d" t "Pitched (naive)",\
     "" u 1:5 every ::1 w linespoints pt 1 lc "#FFCA3A" t "Pitched (precomp.)"

set output "layout_GCC_haswell.tex"
plot "output_GCC_haswell.csv" u 1:2 every ::1 w linespoints pt 1 lc "#b6251c" t "Morton (naive)",\
     "" u 1:3 every ::1 w linespoints pt 1 lc "#149b52" t "Morton (\\texttt{PDEP})",\
     "" u 1:6 every ::1 w linespoints pt 1 lc "#52399d" t "Pitched (naive)",\
     "" u 1:5 every ::1 w linespoints pt 1 lc "#FFCA3A" t "Pitched (precomp.)"
