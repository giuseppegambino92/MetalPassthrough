/* Host-only helpers: pass a file descriptor plus a small message over a Unix socket (SCM_RIGHTS).
 * Used to hand a guest session's socket to a pre-started GPU worker process.
 * Copyright 2026 MetalPassthrough contributors. Apache-2.0. */
#ifndef MPT_FDPASS_H
#define MPT_FDPASS_H
#include <errno.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <unistd.h>

/* Returns 0 on success. */
static inline int mpt_send_fd(int sock, int fd, const void *data, size_t len) {
    struct iovec iov = { (void *)data, len };
    union { struct cmsghdr h; char buf[CMSG_SPACE(sizeof(int))]; } u;
    memset(&u, 0, sizeof u);
    struct msghdr m = { 0 };
    m.msg_iov = &iov;
    m.msg_iovlen = 1;
    m.msg_control = u.buf;
    m.msg_controllen = sizeof u.buf;
    struct cmsghdr *c = CMSG_FIRSTHDR(&m);
    c->cmsg_level = SOL_SOCKET;
    c->cmsg_type = SCM_RIGHTS;
    c->cmsg_len = CMSG_LEN(sizeof(int));
    memcpy(CMSG_DATA(c), &fd, sizeof(int));
    ssize_t n;
    do { n = sendmsg(sock, &m, 0); } while (n < 0 && errno == EINTR);
    return n == (ssize_t)len ? 0 : -1;
}

/* Receives exactly `len` bytes and one descriptor. Returns 0 on success, -1 on error / EOF. */
static inline int mpt_recv_fd(int sock, int *fd, void *data, size_t len) {
    struct iovec iov = { data, len };
    union { struct cmsghdr h; char buf[CMSG_SPACE(sizeof(int))]; } u;
    struct msghdr m = { 0 };
    m.msg_iov = &iov;
    m.msg_iovlen = 1;
    m.msg_control = u.buf;
    m.msg_controllen = sizeof u.buf;
    ssize_t n;
    do { n = recvmsg(sock, &m, MSG_WAITALL); } while (n < 0 && errno == EINTR);
    if (n != (ssize_t)len) return -1;
    struct cmsghdr *c = CMSG_FIRSTHDR(&m);
    if (!c || c->cmsg_level != SOL_SOCKET || c->cmsg_type != SCM_RIGHTS || c->cmsg_len != CMSG_LEN(sizeof(int))) return -1;
    memcpy(fd, CMSG_DATA(c), sizeof(int));
    return 0;
}
#endif
