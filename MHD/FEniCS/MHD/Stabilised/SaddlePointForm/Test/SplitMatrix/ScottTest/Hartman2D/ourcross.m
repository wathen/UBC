function sol = ourcross(a, b)

if length(a) == 2
    a(3) = 0;
    b(3) = 0;
end

sol =[a(2) * b(3) - a(3) * b(2);
      a(3) * b(1) - a(1) * b(3);
      a(1) * b(2) - a(2) * b(1)];